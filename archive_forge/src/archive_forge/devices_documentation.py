from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
List all device states available for a device.

    Up to a maximum of 10 (enforced by service). No pagination.

    Args:
      parent_ref: a Resource reference to a
        cloudiot.projects.locations.registries.devices resource.
      num_states: int, the number of device states to list (max 10).

    Returns:
      List of DeviceStates
    