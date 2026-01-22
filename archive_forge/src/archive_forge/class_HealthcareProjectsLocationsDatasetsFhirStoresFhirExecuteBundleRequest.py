from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirExecuteBundleRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirExecuteBundleRequest
  object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    parent: Name of the FHIR store in which this bundle will be executed.
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    parent = _messages.StringField(2, required=True)