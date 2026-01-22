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
def ValidateCreateDiskFlags(args, enable_snapshots=False, enable_source_snapshot_csek=False, enable_image_csek=False, include_name=True, enable_source_instant_snapshot=False):
    """Validates the values of create-disk related flags."""
    require_csek_key_create = getattr(args, 'require_csek_key_create', None)
    csek_key_file = getattr(args, 'csek_key_file', None)
    resource_names = getattr(args, 'names', [])
    for disk in getattr(args, 'create_disk', []) or []:
        disk_name = disk.get('name')
        if include_name and len(resource_names) > 1 and disk_name:
            raise exceptions.BadArgumentException('--disk', 'Cannot create a disk with [name]={} for more than one instance.'.format(disk_name))
        if disk_name and require_csek_key_create and csek_key_file:
            raise exceptions.BadArgumentException('--disk', 'Cannot create a disk with customer supplied key when disk name is not specified.')
        mode_value = disk.get('mode')
        if mode_value and mode_value not in ('rw', 'ro'):
            raise exceptions.InvalidArgumentException('--disk', 'Value for [mode] in [--disk] must be [rw] or [ro], not [{0}].'.format(mode_value))
        image_value = disk.get('image')
        image_family_value = disk.get('image-family')
        source_snapshot = disk.get('source-snapshot')
        image_csek_file = disk.get('image_csek')
        source_snapshot_csek_file = disk.get('source_snapshot_csek_file')
        source_instant_snapshot = disk.get('source-instant-snapshot')
        disk_source = set()
        if image_value:
            disk_source.add(image_value)
        if image_family_value:
            disk_source.add(image_family_value)
        if source_snapshot:
            disk_source.add(source_snapshot)
        if image_csek_file:
            disk_source.add(image_csek_file)
        if source_snapshot_csek_file:
            disk_source.add(source_snapshot_csek_file)
        if source_instant_snapshot:
            disk_source.add(source_instant_snapshot)
        mutex_attributes = ['[image]', '[image-family]']
        if enable_image_csek:
            mutex_attributes.append('[image-csek-required]')
        if enable_snapshots:
            mutex_attributes.append('[source-snapshot]')
        if enable_source_snapshot_csek:
            mutex_attributes.append('[source-snapshot-csek-required]')
        if enable_source_instant_snapshot:
            mutex_attributes.append('[source-instant-snapshot]')
        formatted_attributes = '{}, or {}'.format(', '.join(mutex_attributes[:-1]), mutex_attributes[-1])
        source_error_message = 'Must specify exactly one of {} for a [--create-disk]. These fields are mutually exclusive.'.format(formatted_attributes)
        if len(disk_source) > 1:
            raise compute_exceptions.ArgumentError(source_error_message)