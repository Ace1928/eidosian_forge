from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansRestoresPatchRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansRestoresPatchRequest object.

  Fields:
    name: Output only. The full name of the Restore resource. Format:
      `projects/*/locations/*/restorePlans/*/restores/*`
    restore: A Restore resource to be passed as the request body.
    updateMask: Optional. This is used to specify the fields to be overwritten
      in the Restore targeted for update. The values for each of these updated
      fields will be taken from the `restore` provided with this request.
      Field names are relative to the root of the resource. If no
      `update_mask` is provided, all fields in `restore` will be written to
      the target Restore resource. Note that OUTPUT_ONLY and IMMUTABLE fields
      in `restore` are ignored and are not used to update the target Restore.
  """
    name = _messages.StringField(1, required=True)
    restore = _messages.MessageField('Restore', 2)
    updateMask = _messages.StringField(3)