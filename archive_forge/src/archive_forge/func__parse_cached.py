import functools
import logging
import matplotlib as mpl
from matplotlib import _api, _mathtext
from matplotlib.ft2font import LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties
from ._mathtext import (  # noqa: reexported API
@functools.lru_cache(50)
def _parse_cached(self, s, dpi, prop, antialiased):
    from matplotlib.backends import backend_agg
    if prop is None:
        prop = FontProperties()
    fontset_class = _api.check_getitem(self._font_type_mapping, fontset=prop.get_math_fontfamily())
    load_glyph_flags = {'vector': LOAD_NO_HINTING, 'raster': backend_agg.get_hinting_flag()}[self._output_type]
    fontset = fontset_class(prop, load_glyph_flags)
    fontsize = prop.get_size_in_points()
    if self._parser is None:
        self.__class__._parser = _mathtext.Parser()
    box = self._parser.parse(s, fontset, fontsize, dpi)
    output = _mathtext.ship(box)
    if self._output_type == 'vector':
        return output.to_vector()
    elif self._output_type == 'raster':
        return output.to_raster(antialiased=antialiased)