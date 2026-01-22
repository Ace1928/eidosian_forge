from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesCloneRequest(_messages.Message):
    """Database instance clone request.

  Fields:
    cloneContext: Contains details about the clone operation.
  """
    cloneContext = _messages.MessageField('CloneContext', 1)