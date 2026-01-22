from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateInstanceConfigMetadata(_messages.Message):
    """Metadata type for the operation returned by CreateInstanceConfig.

  Fields:
    cancelTime: The time at which this operation was cancelled.
    instanceConfig: The target instance config end state.
    progress: The progress of the CreateInstanceConfig operation.
  """
    cancelTime = _messages.StringField(1)
    instanceConfig = _messages.MessageField('InstanceConfig', 2)
    progress = _messages.MessageField('InstanceOperationProgress', 3)