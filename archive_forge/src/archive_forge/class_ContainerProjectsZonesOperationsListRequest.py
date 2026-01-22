from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsZonesOperationsListRequest(_messages.Message):
    """A ContainerProjectsZonesOperationsListRequest object.

  Fields:
    parent: The parent (project and location) where the operations will be
      listed. Specified in the format `projects/*/locations/*`. Location "-"
      matches all zones and all regions.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      parent field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) to return
      operations for, or `-` for all zones. This field has been deprecated and
      replaced by the parent field.
  """
    parent = _messages.StringField(1)
    projectId = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)