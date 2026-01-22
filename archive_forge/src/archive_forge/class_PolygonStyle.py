from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class PolygonStyle(_messages.Message):
    """Represents a PolygonStyle within a StyleSetting

  Fields:
    fillColor: Color of the interior of the polygon in #RRGGBB format.
    fillColorStyler: Column-value, gradient, or bucket styler that is used to
      determine the interior color and opacity of the polygon.
    fillOpacity: Opacity of the interior of the polygon: 0.0 (transparent) to
      1.0 (opaque).
    strokeColor: Color of the polygon border in #RRGGBB format.
    strokeColorStyler: Column-value, gradient or buckets styler that is used
      to determine the border color and opacity.
    strokeOpacity: Opacity of the polygon border: 0.0 (transparent) to 1.0
      (opaque).
    strokeWeight: Width of the polyon border in pixels.
    strokeWeightStyler: Column-value or bucket styler that is used to
      determine the width of the polygon border.
  """
    fillColor = _messages.StringField(1)
    fillColorStyler = _messages.MessageField('StyleFunction', 2)
    fillOpacity = _messages.FloatField(3)
    strokeColor = _messages.StringField(4)
    strokeColorStyler = _messages.MessageField('StyleFunction', 5)
    strokeOpacity = _messages.FloatField(6)
    strokeWeight = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    strokeWeightStyler = _messages.MessageField('StyleFunction', 8)