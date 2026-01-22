from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirCapabilitiesRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirCapabilitiesRequest
  object.

  Fields:
    name: Name of the FHIR store to retrieve the capabilities for.
  """
    name = _messages.StringField(1, required=True)