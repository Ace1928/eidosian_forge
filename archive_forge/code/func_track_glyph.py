from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase
def track_glyph(self, font, glyph):
    """Record that codepoint *glyph* is being typeset using font *font*."""
    self.used.setdefault(font.fname, set()).add(glyph)