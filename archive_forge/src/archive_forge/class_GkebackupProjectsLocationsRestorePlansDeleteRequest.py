from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansDeleteRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansDeleteRequest object.

  Fields:
    etag: Optional. If provided, this value must match the current value of
      the target RestorePlan's etag field or the request is rejected.
    force: Optional. If set to true, any Restores below this RestorePlan will
      also be deleted. Otherwise, the request will only succeed if the
      RestorePlan has no Restores.
    name: Required. Fully qualified RestorePlan name. Format:
      `projects/*/locations/*/restorePlans/*`
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)