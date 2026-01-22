from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityCaaIntelFrontendCustomLevelExplanation(_messages.Message):
    """The Explanation of a Custom Level, which contains the original cel
  expression and the custom level explanation tree NextTAG: 3

  Fields:
    explanation: Custom Level Explanation Tree
    expression: The raw cel expression from customers
  """
    explanation = _messages.MessageField('IdentityCaaIntelFrontendCustomLevelNode', 1)
    expression = _messages.StringField(2)