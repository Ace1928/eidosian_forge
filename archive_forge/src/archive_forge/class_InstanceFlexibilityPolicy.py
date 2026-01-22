from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceFlexibilityPolicy(_messages.Message):
    """Instance flexibility Policy allowing a mixture of VM shapes and
  provisioning models.

  Fields:
    instanceSelectionList: Optional. List of instance selection options that
      the group will use when creating new VMs.
    instanceSelectionResults: Output only. A list of instance selection
      results in the group.
    provisioningModelMix: Optional. Defines how the Group selects the
      provisioning model to ensure required reliability.
  """
    instanceSelectionList = _messages.MessageField('InstanceSelection', 1, repeated=True)
    instanceSelectionResults = _messages.MessageField('InstanceSelectionResult', 2, repeated=True)
    provisioningModelMix = _messages.MessageField('ProvisioningModelMix', 3)