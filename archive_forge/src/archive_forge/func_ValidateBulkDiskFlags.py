from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def ValidateBulkDiskFlags(args, enable_source_snapshot_csek=False, enable_image_csek=False):
    """Validates the values of all disk-related flags."""
    for disk in args.disk or []:
        if 'name' not in disk:
            raise exceptions.InvalidArgumentException('--disk', '[name] is missing in [--disk]. [--disk] value must be of the form [{0}].'.format(instances_flags.DISK_METAVAR))
    instances_flags.ValidateDiskBootFlags(args, enable_kms=True)
    instances_flags.ValidateCreateDiskFlags(args, enable_snapshots=True, enable_source_snapshot_csek=enable_source_snapshot_csek, enable_image_csek=enable_image_csek)