from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaGroupMapping(_messages.Message):
    """A quota group mapping.

  Fields:
    cost: Number of tokens to consume for each request. This allows different
      cost to be associated with different methods that consume from the same
      quota group. By default, each request will cost one token.
    group: The `QuotaGroup.name` of the group. Requests for the mapped methods
      will consume tokens from each of the limits defined in this group.
  """
    cost = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    group = _messages.StringField(2)