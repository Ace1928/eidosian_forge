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
def _MachineTypeNameToCell(machine_type):
    """Returns the formatted name of the given machine type.

  Most machine types will be untouched, with the exception of the custom machine
  type. This modifies the 'custom-N-M' custom machine types with
  'custom (N vCPU, M GiB)'.

  For example, given the following custom machine_type:

    custom-2-3500

  This function will return:

    custom (2 vCPU, 3.41 GiB)

  in the MACHINE_TYPE field when listing out the current instances.

  Args:
    machine_type: The machine type of the given instance

  Returns:
    A formatted version of the given custom machine type (as shown in example
    in docstring above).
  """
    mt = machine_type.get('properties', machine_type).get('machineType')
    if mt:
        return _FormatCustomMachineTypeName(mt)
    return mt