from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedClustersAuthorization(_messages.Message):
    """Configuration related to the cluster RBAC settings.

  Fields:
    adminGroups: Optional. Groups of users that can perform operations as a
      cluster admin. A managed ClusterRoleBinding will be created to grant the
      `cluster-admin` ClusterRole to the groups. Up to ten admin groups can be
      provided. For more info on RBAC, see
      https://kubernetes.io/docs/reference/access-authn-authz/rbac/#user-
      facing-roles
    adminUsers: Optional. Users that can perform operations as a cluster
      admin. A managed ClusterRoleBinding will be created to grant the
      `cluster-admin` ClusterRole to the users. Up to ten admin users can be
      provided. For more info on RBAC, see
      https://kubernetes.io/docs/reference/access-authn-authz/rbac/#user-
      facing-roles
  """
    adminGroups = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedClusterGroup', 1, repeated=True)
    adminUsers = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedClusterUser', 2, repeated=True)