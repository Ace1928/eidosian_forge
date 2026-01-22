from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _FormatCustomMachineTypeName(mt):
    """Checks for custom machine type and modifies output.

  Args:
    mt: machine type to be formatted

  Returns:
    If mt was a custom type, then it will be formatted into the desired custom
      machine type output. Otherwise, it is returned unchanged.

  Helper function for _MachineTypeNameToCell
  """
    custom_family, custom_cpu, custom_ram = instance_utils.GetCpuRamVmFamilyFromCustomName(mt)
    if custom_cpu and custom_ram and custom_family:
        custom_ram_gb = '{0:.2f}'.format(custom_ram / 2 ** 10)
        mt = 'custom ({0}, {1} vCPU, {2} GiB)'.format(custom_family, custom_cpu, custom_ram_gb)
    return mt