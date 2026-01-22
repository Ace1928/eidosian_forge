from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprTypeAbstractType(_messages.Message):
    """Application defined abstract type.

  Fields:
    name: The fully qualified name of this abstract type.
    parameterTypes: Parameter types for this abstract type.
  """
    name = _messages.StringField(1)
    parameterTypes = _messages.MessageField('GoogleApiExprType', 2, repeated=True)