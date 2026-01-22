from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import scheduler
from googlecloudsdk.api_lib import tasks
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import deploy_app_command_util
from googlecloudsdk.api_lib.app import deploy_command_util
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.datastore import index_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.tasks import app_deploy_migration_util
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import create_util
from googlecloudsdk.command_lib.app import deployables
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.command_lib.app import source_files_util
from googlecloudsdk.command_lib.app import staging
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def Deploy(self, service, new_version, code_bucket_ref, image, all_services, gcr_domain, disable_build_cache, wait_for_stop_version, flex_image_build_option=FlexImageBuildOptions.ON_CLIENT, ignore_file=None, service_account=None):
    """Deploy the given service.

    Performs all deployment steps for the given service (if applicable):
    * Enable endpoints (for beta deployments)
    * Build and push the Docker image (Flex only, if image_url not provided)
    * Upload files (non-hermetic deployments and flex deployments with
      flex_image_build_option=FlexImageBuildOptions.ON_SERVER)
    * Create the new version
    * Promote the version to receive all traffic (if --promote given (default))
    * Stop the previous version (if new version promoted and
      --stop-previous-version given (default))

    Args:
      service: deployables.Service, service to be deployed.
      new_version: version_util.Version describing where to deploy the service
      code_bucket_ref: cloud_storage.BucketReference where the service's files
        will be uploaded
      image: str or None, the URL for the Docker image to be deployed (if image
        already exists).
      all_services: dict of service ID to service_util.Service objects
        corresponding to all pre-existing services (used to determine how to
        promote this version to receive all traffic, if applicable).
      gcr_domain: str, Cloud Registry domain, determines the physical location
        of the image. E.g. `us.gcr.io`.
      disable_build_cache: bool, disable the build cache.
      wait_for_stop_version: bool, indicating whether to wait for stop operation
        to finish.
      flex_image_build_option: FlexImageBuildOptions, whether a flex deployment
        should upload files so that the server can build the image or build the
        image on client or build the image on client using the buildpacks.
      ignore_file: custom ignore_file name. Override .gcloudignore file to
        customize files to be skipped.
      service_account: identity this version runs as. If not set, Admin API will
        fallback to use the App Engine default appspot SA.
    """
    log.status.Print('Beginning deployment of service [{service}]...'.format(service=new_version.service))
    if service.service_info.env == env.MANAGED_VMS and flex_image_build_option == FlexImageBuildOptions.ON_SERVER:
        flex_image_build_option = FlexImageBuildOptions.ON_CLIENT
    service_info = service.service_info
    self._ValidateRuntime(service_info)
    source_files = source_files_util.GetSourceFiles(service.upload_dir, service_info.parsed.skip_files.regex, service_info.HasExplicitSkipFiles(), service_info.runtime, service_info.env, service.source, ignore_file=ignore_file)
    build = self._PossiblyBuildAndPush(new_version, service_info, service.upload_dir, source_files, image, code_bucket_ref, gcr_domain, flex_image_build_option)
    manifest = self._PossiblyUploadFiles(image, service_info, service.upload_dir, source_files, code_bucket_ref, flex_image_build_option)
    del source_files
    extra_config_settings = {}
    if disable_build_cache:
        extra_config_settings['no-cache'] = 'true'
    metrics.CustomTimedEvent(metric_names.DEPLOY_API_START)
    self.api_client.DeployService(new_version.service, new_version.id, service_info, manifest, build, extra_config_settings, service_account)
    metrics.CustomTimedEvent(metric_names.DEPLOY_API)
    self._PossiblyPromote(all_services, new_version, wait_for_stop_version)