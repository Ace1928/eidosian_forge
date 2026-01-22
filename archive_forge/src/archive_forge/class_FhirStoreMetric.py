from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FhirStoreMetric(_messages.Message):
    """Count of resources and total storage size by type for a given FHIR
  store.

  Fields:
    count: The total count of FHIR resources in the store of this resource
      type.
    resourceType: The FHIR resource type this metric applies to.
    structuredStorageSizeBytes: The total amount of structured storage used by
      FHIR resources of this resource type in the store.
  """
    count = _messages.IntegerField(1)
    resourceType = _messages.StringField(2)
    structuredStorageSizeBytes = _messages.IntegerField(3)