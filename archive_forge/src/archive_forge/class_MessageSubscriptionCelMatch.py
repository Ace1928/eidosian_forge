from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageSubscriptionCelMatch(_messages.Message):
    """Specifies how to select a route based on a CEL expression.

  Fields:
    checkedExpr: Required. Parsed expression in abstract syntax tree (AST)
      form that has been successfully type checked.
  """
    checkedExpr = _messages.MessageField('GoogleApiExprCheckedExpr', 1)