from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def add_cid_range(self, start, end, glyphs):
    """Add a range to the class by glyph ID. ``start`` and ``end`` are the
        initial and final IDs, and ``glyphs`` is the full list of
        :class:`GlyphName` objects in the range."""
    if self.curr < len(self.glyphs):
        self.original.extend(self.glyphs[self.curr:])
    self.original.append(('\\{}'.format(start), '\\{}'.format(end)))
    self.glyphs.extend(glyphs)
    self.curr = len(self.glyphs)