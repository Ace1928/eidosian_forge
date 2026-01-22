from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
class _ClosureGlyphsT2Decompiler(psCharStrings.SimpleT2Decompiler):

    def __init__(self, components, localSubrs, globalSubrs):
        psCharStrings.SimpleT2Decompiler.__init__(self, localSubrs, globalSubrs)
        self.components = components

    def op_endchar(self, index):
        args = self.popall()
        if len(args) >= 4:
            from fontTools.encodings.StandardEncoding import StandardEncoding
            adx, ady, bchar, achar = args[-4:]
            baseGlyph = StandardEncoding[bchar]
            accentGlyph = StandardEncoding[achar]
            self.components.add(baseGlyph)
            self.components.add(accentGlyph)