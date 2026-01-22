from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsGetRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsGetRequest
  object.

  Fields:
    name: Required. The resource name of the Consent artifact to retrieve.
  """
    name = _messages.StringField(1, required=True)