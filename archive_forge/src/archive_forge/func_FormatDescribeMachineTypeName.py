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
def FormatDescribeMachineTypeName(resources, com_path):
    """Formats a custom machine type when 'instances describe' is called.

  Args:
    resources: dict of resources available for the instance in question
    com_path: command path of the calling command

  Returns:
    If input is a custom type, returns the formatted custom machine type.
      Otherwise, returns None.
  """
    if 'instances' in com_path and 'describe' in com_path:
        if not resources:
            return None
        if 'machineType' not in resources:
            return None
        mt_splitlist = resources['machineType'].split('/')
        mt = mt_splitlist[-1]
        if 'custom' not in mt:
            return None
        formatted_mt = _FormatCustomMachineTypeName(mt)
        mt_splitlist[-1] = formatted_mt
        return '/'.join(mt_splitlist)
    else:
        return None