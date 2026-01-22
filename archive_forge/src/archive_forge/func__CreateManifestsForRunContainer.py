from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import datetime
import io
import os.path
import shutil
import tarfile
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import staging_bucket_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _CreateManifestsForRunContainer(target_to_target_properties, services, from_run_container):
    """Creates manifests for target_id to _TargetProperties object.

  Args:
    target_to_target_properties: map from target_id to _TargetProperties
    services: map of target_id to service_name
    from_run_container: the container to be deployed

  Returns:
    Dictionary of target_id to _TargetProperties where manifest field in
    _TargetProperties is filled in.
  """
    for target_id in target_to_target_properties:
        target_location = target_to_target_properties[target_id].location
        region = target_location.split('/')[-1]
        project = target_location.split('/')[1]
        if target_id not in services:
            raise core_exceptions.Error('Target {} has not been specified in services.'.format(target_id))
        service_name = services[target_id]
        service = cloudrun.ServiceExists(None, project=project, service_name=service_name, region=region, release_track=base.ReleaseTrack.GA)
        if service:
            manifest = resource_projector.MakeSerializable(service)
            manifest = _AddContainerToManifest(manifest, service_name, from_run_container)
            stream_manifest = io.StringIO()
            service_printer = ServicePrinter(stream_manifest)
            service_printer.AddRecord(manifest)
            new_manifest = stream_manifest.getvalue()
            stream_manifest.close()
            target_to_target_properties[target_id].manifest = new_manifest
        else:
            manifest_string = CLOUD_RUN_GENERATED_MANIFEST_TEMPLATE.format(service=service_name, container=from_run_container)
            target_to_target_properties[target_id].manifest = manifest_string
    return target_to_target_properties