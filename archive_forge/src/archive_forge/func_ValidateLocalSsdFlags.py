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
def ValidateLocalSsdFlags(args):
    """Validate local ssd flags."""
    for local_ssd in args.local_ssd or []:
        interface = local_ssd.get('interface')
        if interface and interface not in LOCAL_SSD_INTERFACES:
            raise exceptions.InvalidArgumentException('--local-ssd:interface', 'Unexpected local SSD interface: [{given}]. Legal values are [{ok}].'.format(given=interface, ok=', '.join(LOCAL_SSD_INTERFACES)))
        size = local_ssd.get('size')
        if size is not None:
            if size != constants.SSD_SMALL_PARTITION_GB * constants.BYTES_IN_ONE_GB and size != constants.SSD_LARGE_PARTITION_GB * constants.BYTES_IN_ONE_GB:
                raise exceptions.InvalidArgumentException('--local-ssd:size', 'Unexpected local SSD size: [{given}] bytes. Legal values are {small}GB and {large}GB only.'.format(given=size, small=constants.SSD_SMALL_PARTITION_GB, large=constants.SSD_LARGE_PARTITION_GB))