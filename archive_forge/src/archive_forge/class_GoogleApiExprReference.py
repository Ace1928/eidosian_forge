from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprReference(_messages.Message):
    """Describes a resolved reference to a declaration.

  Fields:
    name: The fully qualified name of the declaration.
    overloadId: For references to functions, this is a list of
      `Overload.overload_id` values which match according to typing rules. If
      the list has more than one element, overload resolution among the
      presented candidates must happen at runtime because of dynamic types.
      The type checker attempts to narrow down this list as much as possible.
      Empty if this is not a reference to a Decl.FunctionDecl.
    value: For references to constants, this may contain the value of the
      constant if known at compile time.
  """
    name = _messages.StringField(1)
    overloadId = _messages.StringField(2, repeated=True)
    value = _messages.MessageField('GoogleApiExprConstant', 3)