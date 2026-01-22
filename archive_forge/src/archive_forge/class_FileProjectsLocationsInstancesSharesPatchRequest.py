from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesSharesPatchRequest(_messages.Message):
    """A FileProjectsLocationsInstancesSharesPatchRequest object.

  Fields:
    name: Output only. The resource name of the share, in the format `projects
      /{project_id}/locations/{location_id}/instances/{instance_id}/shares/{sh
      are_id}`.
    share: A Share resource to be passed as the request body.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include these fields: * "description" * "capacity_gb" * "labels" *
      "nfs_export_options"
  """
    name = _messages.StringField(1, required=True)
    share = _messages.MessageField('Share', 2)
    updateMask = _messages.StringField(3)