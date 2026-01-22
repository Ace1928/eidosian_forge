from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansRestoresGetRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansRestoresGetRequest object.

  Fields:
    name: Required. Name of the restore resource. Format:
      `projects/*/locations/*/restorePlans/*/restores/*`
  """
    name = _messages.StringField(1, required=True)