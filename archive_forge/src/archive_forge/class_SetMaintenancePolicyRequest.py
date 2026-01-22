from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetMaintenancePolicyRequest(_messages.Message):
    """SetMaintenancePolicyRequest sets the maintenance policy for a cluster.

  Fields:
    clusterId: Required. The name of the cluster to update.
    maintenancePolicy: Required. The maintenance policy to be set for the
      cluster. An empty field clears the existing maintenance policy.
    name: The name (project, location, cluster name) of the cluster to set
      maintenance policy. Specified in the format
      `projects/*/locations/*/clusters/*`.
    projectId: Required. The Google Developers Console [project ID or project
      number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects).
    zone: Required. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides.
  """
    clusterId = _messages.StringField(1)
    maintenancePolicy = _messages.MessageField('MaintenancePolicy', 2)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)
    zone = _messages.StringField(5)