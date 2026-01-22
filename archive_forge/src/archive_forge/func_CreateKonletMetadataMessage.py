from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def CreateKonletMetadataMessage(messages, args, instance_name, user_metadata, container_mount_disk_enabled=False, container_mount_disk=None):
    """Helper to create the metadata for konlet."""
    konlet_metadata = {GCE_CONTAINER_DECLARATION: _CreateYamlContainerManifest(args, instance_name, container_mount_disk_enabled=container_mount_disk_enabled, container_mount_disk=container_mount_disk), STACKDRIVER_LOGGING_AGENT_CONFIGURATION: 'true'}
    return metadata_utils.ConstructMetadataMessage(messages, metadata=konlet_metadata, existing_metadata=user_metadata)