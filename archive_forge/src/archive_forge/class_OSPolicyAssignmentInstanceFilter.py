from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyAssignmentInstanceFilter(_messages.Message):
    """Filters to select target VMs for an assignment. If more than one filter
  criteria is specified below, a VM will be selected if and only if it
  satisfies all of them.

  Fields:
    all: Target all VMs in the project. If true, no other criteria is
      permitted.
    exclusionLabels: List of label sets used for VM exclusion. If the list has
      more than one label set, the VM is excluded if any of the label sets are
      applicable for the VM.
    inclusionLabels: List of label sets used for VM inclusion. If the list has
      more than one `LabelSet`, the VM is included if any of the label sets
      are applicable for the VM.
    inventories: List of inventories to select VMs. A VM is selected if its
      inventory data matches at least one of the following inventories.
    osShortNames: Deprecated. Use the `inventories` field instead. A VM is
      selected if it's OS short name matches with any of the values provided
      in this list.
  """
    all = _messages.BooleanField(1)
    exclusionLabels = _messages.MessageField('OSPolicyAssignmentLabelSet', 2, repeated=True)
    inclusionLabels = _messages.MessageField('OSPolicyAssignmentLabelSet', 3, repeated=True)
    inventories = _messages.MessageField('OSPolicyAssignmentInstanceFilterInventory', 4, repeated=True)
    osShortNames = _messages.StringField(5, repeated=True)