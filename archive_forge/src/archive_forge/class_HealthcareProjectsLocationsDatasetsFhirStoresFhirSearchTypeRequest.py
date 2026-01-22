from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirSearchTypeRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirSearchTypeRequest
  object.

  Fields:
    parent: Name of the FHIR store to retrieve resources from.
    resourceType: The FHIR resource type to search, such as Patient or
      Observation. For a complete list, see the FHIR Resource Index ([DSTU2](h
      ttp://hl7.org/implement/standards/fhir/DSTU2/resourcelist.html),
      [STU3](http://hl7.org/implement/standards/fhir/STU3/resourcelist.html),
      [R4](http://hl7.org/implement/standards/fhir/R4/resourcelist.html)).
    searchResourcesRequest: A SearchResourcesRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    resourceType = _messages.StringField(2, required=True)
    searchResourcesRequest = _messages.MessageField('SearchResourcesRequest', 3)