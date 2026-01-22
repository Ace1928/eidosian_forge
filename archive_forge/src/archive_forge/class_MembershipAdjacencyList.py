from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipAdjacencyList(_messages.Message):
    """Membership graph's path information as an adjacency list.

  Fields:
    edges: Each edge contains information about the member that belongs to
      this group. Note: Fields returned here will help identify the specific
      Membership resource (e.g name, preferred_member_key and role), but may
      not be a comprehensive list of all fields.
    group: Resource name of the group that the members belong to.
  """
    edges = _messages.MessageField('Membership', 1, repeated=True)
    group = _messages.StringField(2)