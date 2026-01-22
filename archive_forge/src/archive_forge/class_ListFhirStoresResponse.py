from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFhirStoresResponse(_messages.Message):
    """Lists the FHIR stores in the given dataset.

  Fields:
    fhirStores: The returned FHIR stores. Won't be more FHIR stores than the
      value of page_size in the request.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    fhirStores = _messages.MessageField('FhirStore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)