from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprCall(_messages.Message):
    """A call expression, including calls to predefined functions and
  operators. For example, `value == 10`, `size(map_value)`.

  Fields:
    args: The arguments.
    function: Required. The name of the function or method being called.
    target: The target of an method call-style expression. For example, `x` in
      `x.f()`.
  """
    args = _messages.MessageField('GoogleApiExprExpr', 1, repeated=True)
    function = _messages.StringField(2)
    target = _messages.MessageField('GoogleApiExprExpr', 3)