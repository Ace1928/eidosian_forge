from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BulkMuteFindingsRequest(_messages.Message):
    """Request message for bulk findings update. Note: 1. If multiple bulk
  update requests match the same resource, the order in which they get
  executed is not defined. 2. Once a bulk operation is started, there is no
  way to stop it.

  Fields:
    filter: Expression that identifies findings that should be updated. The
      expression is a list of zero or more restrictions combined via logical
      operators `AND` and `OR`. Parentheses are supported, and `OR` has higher
      precedence than `AND`. Restrictions have the form ` ` and may have a `-`
      character in front of them to indicate negation. The fields map to those
      defined in the corresponding resource. The supported operators are: *
      `=` for all value types. * `>`, `<`, `>=`, `<=` for integer values. *
      `:`, meaning substring matching, for strings. The supported value types
      are: * string literals in quotes. * integer literals without quotes. *
      boolean literals `true` and `false` without quotes.
  """
    filter = _messages.StringField(1)