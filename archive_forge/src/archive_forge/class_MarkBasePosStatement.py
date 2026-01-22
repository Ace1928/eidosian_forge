from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class MarkBasePosStatement(Statement):
    """A mark-to-base positioning rule. The ``base`` should be a
    `glyph-containing object`_. The ``marks`` should be a list of
    (:class:`Anchor`, :class:`MarkClass`) tuples."""

    def __init__(self, base, marks, location=None):
        Statement.__init__(self, location)
        self.base, self.marks = (base, marks)

    def build(self, builder):
        """Calls the builder object's ``add_mark_base_pos`` callback."""
        builder.add_mark_base_pos(self.location, self.base.glyphSet(), self.marks)

    def asFea(self, indent=''):
        res = 'pos base {}'.format(self.base.asFea())
        for a, m in self.marks:
            res += '\n' + indent + SHIFT + '{} mark @{}'.format(a.asFea(), m.name)
        res += ';'
        return res