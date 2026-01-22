from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresCreateRequest object.

  Fields:
    fhirStore: A FhirStore resource to be passed as the request body.
    fhirStoreId: Required. The ID of the FHIR store that is being created. The
      string must match the following regex: `[\\p{L}\\p{N}_\\-\\.]{1,256}`.
    parent: Required. The name of the dataset this FHIR store belongs to.
  """
    fhirStore = _messages.MessageField('FhirStore', 1)
    fhirStoreId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)