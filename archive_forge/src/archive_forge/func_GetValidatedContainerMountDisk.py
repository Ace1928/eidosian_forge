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
def GetValidatedContainerMountDisk(holder, container_mount_disk, disk, create_disk, for_update=False, client=None):
    """Validate --container-mount-disk value."""
    disk = disk or []
    create_disk = create_disk or []
    if not container_mount_disk:
        return
    if not (disk or create_disk or for_update):
        raise exceptions.InvalidArgumentException('--container-mount-disk', 'Must be used with `--disk` or `--create-disk`')
    message = '' if for_update else ' using `--disk` or `--create-disk`.'
    validated_disks = []
    for mount_disk in container_mount_disk:
        if for_update:
            matching_disk, create = _GetMatchingDiskFromMessages(holder, mount_disk.get('name'), disk, client=client)
        else:
            matching_disk, create = _GetMatchingDiskFromFlags(mount_disk.get('name'), disk, create_disk)
        if not mount_disk.get('name'):
            if len(disk + create_disk) != 1:
                raise exceptions.InvalidArgumentException('--container-mount-disk', 'Must specify the name of the disk to be mounted unless exactly one disk is attached to the instance{}.'.format(message))
            name = matching_disk.get('name')
            if not name:
                raise exceptions.InvalidArgumentException('--container-mount-disk', 'When attaching or creating a disk that is also being mounted to a container, must specify the disk name.')
        else:
            name = mount_disk.get('name')
            if not matching_disk:
                raise exceptions.InvalidArgumentException('--container-mount-disk', 'Attempting to mount a disk that is not attached to the instance: must attach a disk named [{}]{}'.format(name, message))
        if matching_disk and matching_disk.get('device_name') and (matching_disk.get('device_name') != matching_disk.get('name')):
            raise exceptions.InvalidArgumentException('--container-mount-disk', 'Container mount disk cannot be used with a device whose device-name is different from its name. The disk with name [{}] has the device-name [{}].'.format(matching_disk.get('name'), matching_disk.get('device_name')))
        mode_value = mount_disk.get('mode')
        if matching_disk:
            _CheckMode(name, mode_value, mount_disk, matching_disk, create)
        if matching_disk and create and mount_disk.get('partition'):
            raise exceptions.InvalidArgumentException('--container-mount-disk', 'Container mount disk cannot specify a partition when the disk is created with --create-disk: disk name [{}], partition [{}]'.format(name, mount_disk.get('partition')))
        mount_disk = copy.deepcopy(mount_disk)
        mount_disk['name'] = mount_disk.get('name') or name
        validated_disks.append(mount_disk)
    return validated_disks