from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresGetRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresGetRequest
  object.

  Fields:
    name: Required. Full name of the VolumeRestore resource. Format:
      `projects/*/locations/*/restorePlans/*/restores/*/volumeRestores/*`
  """
    name = _messages.StringField(1, required=True)