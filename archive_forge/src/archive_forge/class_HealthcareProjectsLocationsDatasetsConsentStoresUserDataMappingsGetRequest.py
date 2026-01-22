from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsGetRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsGetRequest
  object.

  Fields:
    name: Required. The resource name of the User data mapping to retrieve.
  """
    name = _messages.StringField(1, required=True)