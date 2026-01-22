from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsReportsGetRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsReportsGetRequest object.

  Fields:
    name: Required. Name of the resource. The format of this value is as
      follows:
      `organizations/{organization}/locations/{location}/reports/{reportID}`
  """
    name = _messages.StringField(1, required=True)