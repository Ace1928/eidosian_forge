from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansGetRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansGetRequest object.

  Fields:
    name: Required. Fully qualified RestorePlan name. Format:
      `projects/*/locations/*/restorePlans/*`
  """
    name = _messages.StringField(1, required=True)