from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class MarkMarkPosStatement(Statement):
    """A mark-to-mark positioning rule. The ``baseMarks`` must be a
    `glyph-containing object`_. The ``marks`` should be a list of
    (:class:`Anchor`, :class:`MarkClass`) tuples."""

    def __init__(self, baseMarks, marks, location=None):
        Statement.__init__(self, location)
        self.baseMarks, self.marks = (baseMarks, marks)

    def build(self, builder):
        """Calls the builder object's ``add_mark_mark_pos`` callback."""
        builder.add_mark_mark_pos(self.location, self.baseMarks.glyphSet(), self.marks)

    def asFea(self, indent=''):
        res = 'pos mark {}'.format(self.baseMarks.asFea())
        for a, m in self.marks:
            res += '\n' + indent + SHIFT + '{} mark @{}'.format(a.asFea(), m.name)
        res += ';'
        return res