from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetAddonsConfigRequest(_messages.Message):
    """SetAddonsConfigRequest sets the addons associated with the cluster.

  Fields:
    addonsConfig: Required. The desired configurations for the various addons
      available to run in the cluster.
    clusterId: Deprecated. The name of the cluster to upgrade. This field has
      been deprecated and replaced by the name field.
    name: The name (project, location, cluster) of the cluster to set addons.
      Specified in the format `projects/*/locations/*/clusters/*`.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    addonsConfig = _messages.MessageField('AddonsConfig', 1)
    clusterId = _messages.StringField(2)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)
    zone = _messages.StringField(5)