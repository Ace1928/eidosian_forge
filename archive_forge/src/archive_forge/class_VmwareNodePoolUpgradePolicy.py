from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareNodePoolUpgradePolicy(_messages.Message):
    """Parameters that describe the upgrade policy for the node pool.

  Fields:
    independent: Specify the intent to upgrade the node pool with or without
      the control plane upgrade. Defaults to false i.e. upgrade the node pool
      with control plane upgrade. Set this to true to upgrade or downgrade the
      node pool independently from the control plane.
  """
    independent = _messages.BooleanField(1)