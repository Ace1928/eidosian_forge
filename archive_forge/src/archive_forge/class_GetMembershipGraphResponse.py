from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetMembershipGraphResponse(_messages.Message):
    """The response message for MembershipsService.GetMembershipGraph.

  Fields:
    adjacencyList: The membership graph's path information represented as an
      adjacency list.
    groups: The resources representing each group in the adjacency list. Each
      group in this list can be correlated to a 'group' of the
      MembershipAdjacencyList using the 'name' of the Group resource.
  """
    adjacencyList = _messages.MessageField('MembershipAdjacencyList', 1, repeated=True)
    groups = _messages.MessageField('Group', 2, repeated=True)