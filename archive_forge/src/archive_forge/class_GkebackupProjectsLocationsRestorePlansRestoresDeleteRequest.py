from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansRestoresDeleteRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansRestoresDeleteRequest object.

  Fields:
    etag: Optional. If provided, this value must match the current value of
      the target Restore's etag field or the request is rejected.
    force: Optional. If set to true, any VolumeRestores below this restore
      will also be deleted. Otherwise, the request will only succeed if the
      restore has no VolumeRestores.
    name: Required. Full name of the Restore Format:
      `projects/*/locations/*/restorePlans/*/restores/*`
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)