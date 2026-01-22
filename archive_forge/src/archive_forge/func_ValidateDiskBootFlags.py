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
def ValidateDiskBootFlags(args, enable_kms=False):
    """Validates the values of boot disk-related flags."""
    boot_disk_specified = False
    num_of_boot_disk_in_disks = GetNumOfBootDisk(args.disk)
    if num_of_boot_disk_in_disks > 1:
        raise exceptions.BadArgumentException('--disk', 'Each instance can have exactly one boot disk. At least two boot disks were specified through [--disk].')
    num_of_boot_disk_in_create_disks = GetNumOfBootDisk(args.create_disk)
    if num_of_boot_disk_in_create_disks > 1:
        raise exceptions.BadArgumentException('--create-disk', 'Each instance can have exactly one boot disk. At least two boot disks were specified through [--create-disk].')
    if num_of_boot_disk_in_create_disks + num_of_boot_disk_in_disks > 1:
        raise exceptions.BadArgumentException('--create-disk', 'Each instance can have exactly one boot disk. At least two boot disks were specified through [--disk and --create-disk].')
    if num_of_boot_disk_in_create_disks + num_of_boot_disk_in_disks == 1:
        boot_disk_specified = True
    if args.IsSpecified('boot_disk_provisioned_iops'):
        if not args.IsSpecified('boot_disk_type') or not disks_util.IsProvisioningTypeIops(args.boot_disk_type):
            raise exceptions.InvalidArgumentException('--boot-disk-provisioned-iops', '--boot-disk-provisioned-iops cannot be used with the given disk type.')
    if args.IsSpecified('boot_disk_provisioned_throughput'):
        if not args.IsSpecified('boot_disk_type') or not disks_util.IsProvisioningTypeThroughput(args.boot_disk_type):
            raise exceptions.InvalidArgumentException('--boot-disk-provisioned-throughput', '--boot-disk-provisioned-throughput cannot be used with the given disk type.')
    if args.IsSpecified('boot_disk_size'):
        size_gb = utils.BytesToGb(args.boot_disk_size)
        if args.IsSpecified('boot_disk_type') and args.boot_disk_type in constants.LEGACY_DISK_TYPE_LIST and (size_gb < 10):
            raise exceptions.InvalidArgumentException('--boot-disk-size', 'Value must be greater than or equal to 10 GB; reveived {0} GB'.format(size_gb))
    if args.image and boot_disk_specified:
        raise exceptions.BadArgumentException('--disk', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk] and another through [--image].')
    if boot_disk_specified:
        if args.boot_disk_device_name:
            raise exceptions.BadArgumentException('--boot-disk-device-name', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
        if args.boot_disk_type:
            raise exceptions.BadArgumentException('--boot-disk-type', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
        if args.boot_disk_size:
            raise exceptions.BadArgumentException('--boot-disk-size', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
        if not args.boot_disk_auto_delete:
            raise exceptions.BadArgumentException('--no-boot-disk-auto-delete', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
        if enable_kms:
            if args.boot_disk_kms_key:
                raise exceptions.BadArgumentException('--boot-disk-kms-key', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
            if args.boot_disk_kms_keyring:
                raise exceptions.BadArgumentException('--boot-disk-kms-keyring', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
            if args.boot_disk_kms_location:
                raise exceptions.BadArgumentException('--boot-disk-kms-location', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')
            if args.boot_disk_kms_project:
                raise exceptions.BadArgumentException('--boot-disk-kms-project', 'Each instance can have exactly one boot disk. One boot disk was specified through [--disk or --create-disk]')