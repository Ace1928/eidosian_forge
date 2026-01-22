from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresQueryAccessibleDataRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsConsentStoresQueryAccessibleDataRequest
  object.

  Fields:
    consentStore: Required. Name of the consent store to retrieve User data
      mappings from.
    queryAccessibleDataRequest: A QueryAccessibleDataRequest resource to be
      passed as the request body.
  """
    consentStore = _messages.StringField(1, required=True)
    queryAccessibleDataRequest = _messages.MessageField('QueryAccessibleDataRequest', 2)