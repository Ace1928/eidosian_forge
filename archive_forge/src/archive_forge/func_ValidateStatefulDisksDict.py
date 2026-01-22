from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def ValidateStatefulDisksDict(stateful_disks, flag_name):
    """Validate device-name and auto-delete flags in a stateful disk."""
    device_names = set()
    for stateful_disk in stateful_disks or []:
        if not stateful_disk.get('device-name'):
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='[device-name] is required')
        if stateful_disk.get('device-name') in device_names:
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='[device-name] `{0}` is not unique in the collection'.format(stateful_disk.get('device-name')))
        device_names.add(stateful_disk.get('device-name'))