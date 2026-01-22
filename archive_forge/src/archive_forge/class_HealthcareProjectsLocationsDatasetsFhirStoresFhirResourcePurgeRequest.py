from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirResourcePurgeRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirResourcePurgeRequest
  object.

  Fields:
    name: The name of the resource to purge.
  """
    name = _messages.StringField(1, required=True)