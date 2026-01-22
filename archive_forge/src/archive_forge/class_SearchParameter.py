from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchParameter(_messages.Message):
    """Contains the versioned name and the URL for one SearchParameter.

  Fields:
    canonicalUrl: The canonical url of the search parameter resource.
    parameter: The versioned name of the search parameter resource. The format
      is projects/{project-id}/locations/{location}/datasets/{dataset-
      id}/fhirStores/{fhirStore-id}/fhir/SearchParameter/{resource-
      id}/_history/{version-id} For fhir stores with
      disable_resource_versioning=true, the format is projects/{project-
      id}/locations/{location}/datasets/{dataset-id}/fhirStores/{fhirStore-
      id}/fhir/SearchParameter/{resource-id}/
  """
    canonicalUrl = _messages.StringField(1)
    parameter = _messages.StringField(2)