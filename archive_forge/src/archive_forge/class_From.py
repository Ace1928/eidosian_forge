from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class From(_messages.Message):
    """From clause specifies the principals to whom the rule applies.

  Fields:
    principals: List of requesting principal identifiers.
  """
    principals = _messages.StringField(1, repeated=True)