from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InfoTypeConfig(_messages.Message):
    """Specifies how to use infoTypes for evaluation. For example, a user might
  only want to evaluate `PERSON`, `LOCATION`, and `AGE`.

  Fields:
    evaluateList: A FilterList attribute.
    ignoreList: A FilterList attribute.
    strictMatching: If `TRUE`, infoTypes described by `filter` are used for
      evaluation. Otherwise, infoTypes are not considered for evaluation. For
      example: * Annotated text: "Toronto is a location" * Finding 1:
      `{"infoType": "PERSON", "quote": "Toronto", "start": 0, "end": 7}` *
      Finding 2: `{"infoType": "CITY", "quote": "Toronto", "start": 0, "end":
      7}` * Finding 3: `{}` * Ground truth: `{"infoType": "LOCATION", "quote":
      "Toronto", "start": 0, "end": 7}` When `strict_matching` is `TRUE`: *
      Finding 1: 1 false positive * Finding 2: 1 false positive * Finding 3: 1
      false negative When `strict_matching` is `FALSE`: * Finding 1: 1 true
      positive * Finding 2: 1 true positive * Finding 3: 1 false negative
  """
    evaluateList = _messages.MessageField('FilterList', 1)
    ignoreList = _messages.MessageField('FilterList', 2)
    strictMatching = _messages.BooleanField(3)