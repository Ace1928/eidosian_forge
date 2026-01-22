from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class GradientValue(_messages.Message):
    """Gradient function that interpolates a range of colors based on column
    value.

    Messages:
      ColorsValueListEntry: A ColorsValueListEntry object.

    Fields:
      colors: Array with two or more colors.
      max: Higher-end of the interpolation range: rows with this value will be
        assigned to colors[n-1].
      min: Lower-end of the interpolation range: rows with this value will be
        assigned to colors[0].
    """

    class ColorsValueListEntry(_messages.Message):
        """A ColorsValueListEntry object.

      Fields:
        color: Color in #RRGGBB format.
        opacity: Opacity of the color: 0.0 (transparent) to 1.0 (opaque).
      """
        color = _messages.StringField(1)
        opacity = _messages.FloatField(2)
    colors = _messages.MessageField('ColorsValueListEntry', 1, repeated=True)
    max = _messages.FloatField(2)
    min = _messages.FloatField(3)