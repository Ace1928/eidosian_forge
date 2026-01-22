from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessOptionsIndividualPageSelector(_messages.Message):
    """A list of individual page numbers.

  Fields:
    pages: Optional. Indices of the pages (starting from 1).
  """
    pages = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)