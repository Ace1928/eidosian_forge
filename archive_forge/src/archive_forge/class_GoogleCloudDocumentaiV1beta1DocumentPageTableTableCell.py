from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageTableTableCell(_messages.Message):
    """A cell representation inside the table.

  Fields:
    colSpan: How many columns this cell spans.
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for TableCell.
    rowSpan: How many rows this cell spans.
  """
    colSpan = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageDetectedLanguage', 2, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageLayout', 3)
    rowSpan = _messages.IntegerField(4, variant=_messages.Variant.INT32)