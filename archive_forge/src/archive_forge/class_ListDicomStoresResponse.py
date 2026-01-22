from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDicomStoresResponse(_messages.Message):
    """Lists the DICOM stores in the given dataset.

  Fields:
    dicomStores: The returned DICOM stores. Won't be more DICOM stores than
      the value of page_size in the request.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    dicomStores = _messages.MessageField('DicomStore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)