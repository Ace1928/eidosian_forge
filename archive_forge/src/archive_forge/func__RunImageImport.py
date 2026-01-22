from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
def _RunImageImport(self, args, import_args, tags, output_filter):
    image_tag = daisy_utils.GetDefaultBuilderVersion()
    if hasattr(args, 'docker_image_tag'):
        image_tag = args.docker_image_tag
    if _HasExternalCloudProvider(args):
        return daisy_utils.RunOnestepImageImport(args, import_args, tags, _OUTPUT_FILTER, release_track=self.ReleaseTrack().id.lower() if self.ReleaseTrack() else None, docker_image_tag=image_tag)
    return daisy_utils.RunImageImport(args, import_args, tags, _OUTPUT_FILTER, release_track=self.ReleaseTrack().id.lower() if self.ReleaseTrack() else None, docker_image_tag=image_tag)