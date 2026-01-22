from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunJobConfig(_messages.Message):
    """Message for Cloud Run job configs.

  Fields:
    bindings: Bindings to other resources.
    config: Configuration for the job.
  """
    bindings = _messages.MessageField('ServiceResourceBindingConfig', 1, repeated=True)
    config = _messages.MessageField('JobSettingsConfig', 2)