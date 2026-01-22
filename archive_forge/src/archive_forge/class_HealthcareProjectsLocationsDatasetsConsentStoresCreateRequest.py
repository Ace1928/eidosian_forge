from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresCreateRequest object.

  Fields:
    consentStore: A ConsentStore resource to be passed as the request body.
    consentStoreId: Required. The ID of the consent store to create. The
      string must match the following regex: `[\\p{L}\\p{N}_\\-\\.]{1,256}`.
      Cannot be changed after creation.
    parent: Required. The name of the dataset this consent store belongs to.
  """
    consentStore = _messages.MessageField('ConsentStore', 1)
    consentStoreId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)