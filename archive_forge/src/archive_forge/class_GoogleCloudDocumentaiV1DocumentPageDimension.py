from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageDimension(_messages.Message):
    """Dimension for the page.

  Fields:
    height: Page height.
    unit: Dimension unit.
    width: Page width.
  """
    height = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    unit = _messages.StringField(2)
    width = _messages.FloatField(3, variant=_messages.Variant.FLOAT)