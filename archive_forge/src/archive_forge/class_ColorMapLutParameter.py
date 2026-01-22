from ... import colormap
from ...widgets.ColorMapButton import ColorMapButton
from .basetypes import Parameter, WidgetParameterItem
class ColorMapLutParameter(Parameter):
    itemClass = ColorMapLutParameterItem

    def _interpretValue(self, v):
        if isinstance(v, str):
            v = colormap.get(v)
        if v is not None and (not isinstance(v, colormap.ColorMap)):
            raise TypeError('Cannot set colormap parameter from object %r' % v)
        return v