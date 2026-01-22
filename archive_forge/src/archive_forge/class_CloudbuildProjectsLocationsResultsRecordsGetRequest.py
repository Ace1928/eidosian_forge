from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsResultsRecordsGetRequest(_messages.Message):
    """A CloudbuildProjectsLocationsResultsRecordsGetRequest object.

  Fields:
    name: Required. The name of the Record to retrieve. Format:
      projects/{project}/locations/{location}/results/{result}/records/{record
      }
  """
    name = _messages.StringField(1, required=True)