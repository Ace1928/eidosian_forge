from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AsyncOptions(_messages.Message):
    """Async options that determine when a resource should finish.

  Fields:
    methodMatch: Method regex where this policy will apply.
    pollingOptions: Deployment manager will poll instances for this API
      resource setting a RUNNING state, and blocking until polling conditions
      tell whether the resource is completed or failed.
  """
    methodMatch = _messages.StringField(1)
    pollingOptions = _messages.MessageField('PollingOptions', 2)