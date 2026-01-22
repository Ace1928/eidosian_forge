from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def add_class(self, gc):
    """Add glyphs from the given :class:`GlyphClassName` object to the
        class."""
    if self.curr < len(self.glyphs):
        self.original.extend(self.glyphs[self.curr:])
    self.original.append(gc)
    self.glyphs.extend(gc.glyphSet())
    self.curr = len(self.glyphs)