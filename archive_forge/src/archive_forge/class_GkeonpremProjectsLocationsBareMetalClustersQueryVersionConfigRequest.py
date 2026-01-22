from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersQueryVersionConfigRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalClustersQueryVersionConfigRequest
  object.

  Fields:
    createConfig_adminClusterMembership: The admin cluster membership. This is
      the full resource name of the admin cluster's fleet membership. Format:
      "projects/{project}/locations/{location}/memberships/{membership}"
    createConfig_adminClusterName: The admin cluster resource name. This is
      the full resource name of the admin cluster resource. Format: "projects/
      {project}/locations/{location}/bareMetalAdminClusters/{bare_metal_admin_
      cluster}"
    parent: Required. The parent of the project and location to query for
      version config. Format: "projects/{project}/locations/{location}"
    upgradeConfig_clusterName: The user cluster resource name. This is the
      full resource name of the user cluster resource. Format: "projects/{proj
      ect}/locations/{location}/bareMetalClusters/{bare_metal_cluster}"
  """
    createConfig_adminClusterMembership = _messages.StringField(1)
    createConfig_adminClusterName = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    upgradeConfig_clusterName = _messages.StringField(4)