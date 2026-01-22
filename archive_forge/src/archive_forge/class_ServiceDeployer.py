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
class ServiceDeployer(object):
    """Coordinator (reusable) for deployment of one service at a time.

  Attributes:
    api_client: api_lib.app.appengine_api_client.AppengineClient, App Engine
      Admin API client.
    deploy_options: DeployOptions, the options to use for services deployed by
      this ServiceDeployer.
  """

    def __init__(self, api_client, deploy_options):
        self.api_client = api_client
        self.deploy_options = deploy_options

    def _ValidateRuntime(self, service_info):
        """Validates explicit runtime builders are not used without the feature on.

    Args:
      service_info: yaml_parsing.ServiceYamlInfo, service configuration to be
        deployed

    Raises:
      InvalidRuntimeNameError: if the runtime name is invalid for the deployment
        (see above).
    """
        runtime = service_info.runtime
        if runtime == 'custom':
            return
        needs_dockerfile = True
        strategy = self.deploy_options.runtime_builder_strategy
        use_runtime_builders = deploy_command_util.ShouldUseRuntimeBuilders(service_info, strategy, needs_dockerfile)
        if not use_runtime_builders and (not ORIGINAL_RUNTIME_RE.match(runtime)):
            raise InvalidRuntimeNameError(runtime, ORIGINAL_RUNTIME_RE_STRING)

    def _PossiblyBuildAndPush(self, new_version, service, upload_dir, source_files, image, code_bucket_ref, gcr_domain, flex_image_build_option):
        """Builds and Pushes the Docker image if necessary for this service.

    Args:
      new_version: version_util.Version describing where to deploy the service
      service: yaml_parsing.ServiceYamlInfo, service configuration to be
        deployed
      upload_dir: str, path to the service's upload directory
      source_files: [str], relative paths to upload.
      image: str or None, the URL for the Docker image to be deployed (if image
        already exists).
      code_bucket_ref: cloud_storage.BucketReference where the service's files
        have been uploaded
      gcr_domain: str, Cloud Registry domain, determines the physical location
        of the image. E.g. `us.gcr.io`.
      flex_image_build_option: FlexImageBuildOptions, whether a flex deployment
        should upload files so that the server can build the image or build the
        image on client or build the image on client using the buildpacks.

    Returns:
      BuildArtifact, a wrapper which contains either the build ID for
        an in-progress build, or the name of the container image for a serial
        build. Possibly None if the service does not require an image.
    Raises:
      RequiredFileMissingError: if a required file is not uploaded.
    """
        build = None
        if image:
            if service.RequiresImage() and service.parsed.skip_files.regex:
                log.warning('Deployment of service [{0}] will ignore the skip_files field in the configuration file, because the image has already been built.'.format(new_version.service))
            return app_cloud_build.BuildArtifact.MakeImageArtifact(image)
        elif service.RequiresImage():
            if not _AppYamlInSourceFiles(source_files, service.GetAppYamlBasename()):
                raise RequiredFileMissingError(service.GetAppYamlBasename())
            if flex_image_build_option == FlexImageBuildOptions.ON_SERVER:
                cloud_build_options = {'appYamlPath': service.GetAppYamlBasename()}
                timeout = properties.VALUES.app.cloud_build_timeout.Get()
                if timeout:
                    build_timeout = int(times.ParseDuration(timeout, default_suffix='s').total_seconds)
                    cloud_build_options['cloudBuildTimeout'] = six.text_type(build_timeout) + 's'
                build = app_cloud_build.BuildArtifact.MakeBuildOptionsArtifact(cloud_build_options)
            else:
                build = deploy_command_util.BuildAndPushDockerImage(new_version.project, service, upload_dir, source_files, new_version.id, code_bucket_ref, gcr_domain, self.deploy_options.runtime_builder_strategy, self.deploy_options.parallel_build, flex_image_build_option == FlexImageBuildOptions.BUILDPACK_ON_CLIENT)
        return build

    def _PossiblyPromote(self, all_services, new_version, wait_for_stop_version):
        """Promotes the new version to default (if specified by the user).

    Args:
      all_services: dict of service ID to service_util.Service objects
        corresponding to all pre-existing services (used to determine how to
        promote this version to receive all traffic, if applicable).
      new_version: version_util.Version describing where to deploy the service
      wait_for_stop_version: bool, indicating whether to wait for stop operation
        to finish.

    Raises:
      VersionPromotionError: if the version could not successfully promoted
    """
        if self.deploy_options.promote:
            try:
                version_util.PromoteVersion(all_services, new_version, self.api_client, self.deploy_options.stop_previous_version, wait_for_stop_version)
            except apitools_exceptions.HttpError as err:
                err_str = six.text_type(core_api_exceptions.HttpException(err))
                raise VersionPromotionError(err_str)
        elif self.deploy_options.stop_previous_version:
            log.info('Not stopping previous version because new version was not promoted.')

    def _PossiblyUploadFiles(self, image, service_info, upload_dir, source_files, code_bucket_ref, flex_image_build_option):
        """Uploads files for this deployment is required for this service.

    Uploads if flex_image_build_option is FlexImageBuildOptions.ON_SERVER,
    or if the deployment is non-hermetic and the image is not provided.

    Args:
      image: str or None, the URL for the Docker image to be deployed (if image
        already exists).
      service_info: yaml_parsing.ServiceYamlInfo, service configuration to be
        deployed
      upload_dir: str, path to the service's upload directory
      source_files: [str], relative paths to upload.
      code_bucket_ref: cloud_storage.BucketReference where the service's files
        have been uploaded
      flex_image_build_option: FlexImageBuildOptions, whether a flex deployment
        should upload files so that the server can build the image or build the
        image on client or build the image on client using the buildpacks.

    Returns:
      Dictionary mapping source files to Google Cloud Storage locations.

    Raises:
      RequiredFileMissingError: if a required file is not uploaded.
    """
        manifest = None
        if not image and (flex_image_build_option == FlexImageBuildOptions.ON_SERVER or not service_info.is_hermetic):
            if service_info.env == env.FLEX and (not _AppYamlInSourceFiles(source_files, service_info.GetAppYamlBasename())):
                raise RequiredFileMissingError(service_info.GetAppYamlBasename())
            limit = None
            if service_info.env == env.STANDARD and service_info.runtime in _RUNTIMES_WITH_FILE_SIZE_LIMITS:
                limit = _MAX_FILE_SIZE_STANDARD
            manifest = deploy_app_command_util.CopyFilesToCodeBucket(upload_dir, source_files, code_bucket_ref, max_file_size=limit)
        return manifest

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