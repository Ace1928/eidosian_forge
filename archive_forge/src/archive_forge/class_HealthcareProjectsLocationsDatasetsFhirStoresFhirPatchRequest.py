from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirPatchRequest object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    name: The name of the resource to update.
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    name = _messages.StringField(2, required=True)