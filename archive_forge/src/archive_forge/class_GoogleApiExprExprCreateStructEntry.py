from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprCreateStructEntry(_messages.Message):
    """Represents an entry.

  Fields:
    fieldKey: The field key for a message creator statement.
    id: Required. An id assigned to this node by the parser which is unique in
      a given expression tree. This is used to associate type information and
      other attributes to the node.
    mapKey: The key expression for a map creation statement.
    optionalEntry: Whether the key-value pair is optional.
    value: Required. The value assigned to the key. If the optional_entry
      field is true, the expression must resolve to an optional-typed value.
      If the optional value is present, the key will be set; however, if the
      optional value is absent, the key will be unset.
  """
    fieldKey = _messages.StringField(1)
    id = _messages.IntegerField(2)
    mapKey = _messages.MessageField('GoogleApiExprExpr', 3)
    optionalEntry = _messages.BooleanField(4)
    value = _messages.MessageField('GoogleApiExprExpr', 5)