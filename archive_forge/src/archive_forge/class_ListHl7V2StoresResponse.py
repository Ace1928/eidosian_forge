from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListHl7V2StoresResponse(_messages.Message):
    """Lists the HL7v2 stores in the given dataset.

  Fields:
    hl7V2Stores: The returned HL7v2 stores. Won't be more HL7v2 stores than
      the value of page_size in the request.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    hl7V2Stores = _messages.MessageField('Hl7V2Store', 1, repeated=True)
    nextPageToken = _messages.StringField(2)