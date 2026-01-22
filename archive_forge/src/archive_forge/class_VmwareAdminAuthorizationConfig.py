from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminAuthorizationConfig(_messages.Message):
    """VmwareAdminAuthorizationConfig represents configuration for admin
  cluster authorization.

  Fields:
    viewerUsers: For VMware admin clusters, users will be granted the cluster-
      viewer role on the cluster.
  """
    viewerUsers = _messages.MessageField('ClusterUser', 1, repeated=True)