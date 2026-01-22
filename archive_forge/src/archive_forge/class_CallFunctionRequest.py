from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CallFunctionRequest(_messages.Message):
    """Request for the `CallFunction` method.

  Fields:
    data: Required. Input to be passed to the function.
  """
    data = _messages.StringField(1)