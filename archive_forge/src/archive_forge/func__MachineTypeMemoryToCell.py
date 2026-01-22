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
def _MachineTypeMemoryToCell(machine_type):
    """Returns the memory of the given machine type in GB."""
    memory = machine_type.get('memoryMb')
    if memory:
        return '{0:5.2f}'.format(float(memory) / 2 ** 10)
    else:
        return ''