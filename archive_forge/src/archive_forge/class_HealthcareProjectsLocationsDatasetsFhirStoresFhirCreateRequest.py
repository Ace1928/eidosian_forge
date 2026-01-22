from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirCreateRequest object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    parent: The name of the FHIR store this resource belongs to.
    type: The FHIR resource type to create, such as Patient or Observation.
      For a complete list, see the FHIR Resource Index ([DSTU2](http://hl7.org
      /implement/standards/fhir/DSTU2/resourcelist.html),
      [STU3](http://hl7.org/implement/standards/fhir/STU3/resourcelist.html),
      [R4](http://hl7.org/implement/standards/fhir/R4/resourcelist.html)).
      Must match the resource type in the provided content.
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    parent = _messages.StringField(2, required=True)
    type = _messages.StringField(3, required=True)