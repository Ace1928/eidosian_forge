from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _ValidateNFSMountPermissions(mount_permissions_input):
    """Validates NFS mount permissions field, throws exception if invalid."""
    mount_permissions = mount_permissions_input.upper()
    if mount_permissions not in NFS_MOUNT_PERMISSIONS_CHOICES:
        raise exceptions.InvalidArgumentException('--allowed-client', 'Invalid value {} for mount-permissions'.format(mount_permissions_input))
    return mount_permissions