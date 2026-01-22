from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MacroCallsValue(_messages.Message):
    """A map from the parse node id where a macro replacement was made to the
    call `Expr` that resulted in a macro expansion. For example,
    `has(value.field)` is a function call that is replaced by a `test_only`
    field selection in the AST. Likewise, the call `list.exists(e, e > 10)`
    translates to a comprehension expression. The key in the map corresponds
    to the expression id of the expanded macro, and the value is the call
    `Expr` that was replaced.

    Messages:
      AdditionalProperty: An additional property for a MacroCallsValue object.

    Fields:
      additionalProperties: Additional properties of type MacroCallsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MacroCallsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiExprExpr attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleApiExprExpr', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)