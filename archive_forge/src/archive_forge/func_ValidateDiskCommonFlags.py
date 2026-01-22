from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def ValidateDiskCommonFlags(args):
    """Validates the values of common disk-related flags."""
    for disk in args.disk or []:
        disk_name = disk.get('name')
        if not disk_name:
            raise exceptions.InvalidArgumentException('--disk', '[name] is missing in [--disk]. [--disk] value must be of the form [{0}].'.format(DISK_METAVAR))
        mode_value = disk.get('mode')
        if mode_value and mode_value not in ('rw', 'ro'):
            raise exceptions.InvalidArgumentException('--disk', 'Value for [mode] in [--disk] must be [rw] or [ro], not [{0}].'.format(mode_value))