from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentStyleFontSize(_messages.Message):
    """Font size with unit.

  Fields:
    size: Font size for the text.
    unit: Unit for the font size. Follows CSS naming (such as `in`, `px`, and
      `pt`).
  """
    size = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    unit = _messages.StringField(2)