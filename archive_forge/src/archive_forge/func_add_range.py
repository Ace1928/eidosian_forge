from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def add_range(self, start, end, glyphs):
    """Add a range (e.g. ``A-Z``) to the class. ``start`` and ``end``
        are either :class:`GlyphName` objects or strings representing the
        start and end glyphs in the class, and ``glyphs`` is the full list of
        :class:`GlyphName` objects in the range."""
    if self.curr < len(self.glyphs):
        self.original.extend(self.glyphs[self.curr:])
    self.original.append((start, end))
    self.glyphs.extend(glyphs)
    self.curr = len(self.glyphs)