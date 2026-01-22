from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsProjectDataProfilesGetRequest(_messages.Message):
    """A DlpProjectsLocationsProjectDataProfilesGetRequest object.

  Fields:
    name: Required. Resource name, for example
      `organizations/12345/locations/us/projectDataProfiles/53234423`.
  """
    name = _messages.StringField(1, required=True)