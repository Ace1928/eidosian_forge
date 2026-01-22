from types import MethodType
from os.path import isfile
from kivy.resources import resource_find
from kivy.core.text import LabelBase, FontContextManagerBase
from kivy.core.text._text_pango import (
class LabelPango(LabelBase):
    _font_family_support = True

    def __init__(self, *largs, **kwargs):
        self.get_extents = MethodType(kpango_get_extents, self)
        self.get_ascent = MethodType(kpango_get_ascent, self)
        self.get_descent = MethodType(kpango_get_descent, self)
        super(LabelPango, self).__init__(*largs, **kwargs)
    find_base_direction = staticmethod(kpango_find_base_dir)

    def _render_begin(self):
        self._rdr = KivyPangoRenderer(*self._size)

    def _render_text(self, text, x, y):
        self._rdr.render(self, text, x, y)

    def _render_end(self):
        imgdata = self._rdr.get_ImageData()
        del self._rdr
        return imgdata