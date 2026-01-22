from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprTypeListType(_messages.Message):
    """List type with typed elements, e.g. `list`.

  Fields:
    elemType: The element type.
  """
    elemType = _messages.MessageField('GoogleApiExprType', 1)