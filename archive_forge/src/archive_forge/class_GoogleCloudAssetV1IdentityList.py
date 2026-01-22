from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1IdentityList(_messages.Message):
    """The identities and group edges.

  Fields:
    groupEdges: Group identity edges of the graph starting from the binding's
      group members to any node of the identities. The Edge.source_node
      contains a group, such as `group:parent@google.com`. The
      Edge.target_node contains a member of the group, such as
      `group:child@google.com` or `user:foo@google.com`. This field is present
      only if the output_group_edges option is enabled in request.
    identities: Only the identities that match one of the following conditions
      will be presented: - The identity_selector, if it is specified in
      request; - Otherwise, identities reachable from the policy binding's
      members.
  """
    groupEdges = _messages.MessageField('GoogleCloudAssetV1Edge', 1, repeated=True)
    identities = _messages.MessageField('GoogleCloudAssetV1Identity', 2, repeated=True)