from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsGetRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsGetRequest
  object.

  Fields:
    name: Required. The name of the CutoverJob.
  """
    name = _messages.StringField(1, required=True)