from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstanceConfigMetadata(_messages.Message):
    """Metadata type for the operation returned by UpdateInstanceConfig.

  Fields:
    cancelTime: The time at which this operation was cancelled.
    instanceConfig: The desired instance config after updating.
    progress: The progress of the UpdateInstanceConfig operation.
  """
    cancelTime = _messages.StringField(1)
    instanceConfig = _messages.MessageField('InstanceConfig', 2)
    progress = _messages.MessageField('InstanceOperationProgress', 3)