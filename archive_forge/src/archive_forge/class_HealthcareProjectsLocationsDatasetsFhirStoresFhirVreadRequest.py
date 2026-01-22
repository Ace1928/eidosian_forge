from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirVreadRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirVreadRequest object.

  Fields:
    name: The name of the resource version to retrieve.
  """
    name = _messages.StringField(1, required=True)