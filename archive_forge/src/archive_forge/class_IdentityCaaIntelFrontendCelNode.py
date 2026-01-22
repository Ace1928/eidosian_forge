from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityCaaIntelFrontendCelNode(_messages.Message):
    """Cel node, including evaluation results and metadata NextTAG: 7

  Fields:
    endPosition: Optional, it exists if it is CustomLevel Access Level. End
      position of an expression in the original condition, by character, end
      included
    nodeValues: Repeated as one node id may correspond to multiple evaluation
      values. e.g.in comprehension expr, [1,2,3].all(x, x > 0), call_expr
      "_>_" has 3 values corresponding to the evaluation of list values
      individually sequentially
    startPosition: Optional, it exists if it is CustomLevel Access Level.
      Start position of an expression in the original condition, by character
  """
    endPosition = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    nodeValues = _messages.MessageField('IdentityCaaIntelFrontendNodeValue', 2, repeated=True)
    startPosition = _messages.IntegerField(3, variant=_messages.Variant.INT32)