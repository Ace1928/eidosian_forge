from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class PairPosStatement(Statement):
    """A pair positioning statement.

    ``glyphs1`` and ``glyphs2`` should be `glyph-containing objects`_.
    ``valuerecord1`` should be a :class:`ValueRecord` object;
    ``valuerecord2`` should be either a :class:`ValueRecord` object or ``None``.
    If ``enumerated`` is true, then this is expressed as an
    `enumerated pair <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#6.b.ii>`_.
    """

    def __init__(self, glyphs1, valuerecord1, glyphs2, valuerecord2, enumerated=False, location=None):
        Statement.__init__(self, location)
        self.enumerated = enumerated
        self.glyphs1, self.valuerecord1 = (glyphs1, valuerecord1)
        self.glyphs2, self.valuerecord2 = (glyphs2, valuerecord2)

    def build(self, builder):
        """Calls a callback on the builder object:

        * If the rule is enumerated, calls ``add_specific_pair_pos`` on each
          combination of first and second glyphs.
        * If the glyphs are both single :class:`GlyphName` objects, calls
          ``add_specific_pair_pos``.
        * Else, calls ``add_class_pair_pos``.
        """
        if self.enumerated:
            g = [self.glyphs1.glyphSet(), self.glyphs2.glyphSet()]
            seen_pair = False
            for glyph1, glyph2 in itertools.product(*g):
                seen_pair = True
                builder.add_specific_pair_pos(self.location, glyph1, self.valuerecord1, glyph2, self.valuerecord2)
            if not seen_pair:
                raise FeatureLibError('Empty glyph class in positioning rule', self.location)
            return
        is_specific = isinstance(self.glyphs1, GlyphName) and isinstance(self.glyphs2, GlyphName)
        if is_specific:
            builder.add_specific_pair_pos(self.location, self.glyphs1.glyph, self.valuerecord1, self.glyphs2.glyph, self.valuerecord2)
        else:
            builder.add_class_pair_pos(self.location, self.glyphs1.glyphSet(), self.valuerecord1, self.glyphs2.glyphSet(), self.valuerecord2)

    def asFea(self, indent=''):
        res = 'enum ' if self.enumerated else ''
        if self.valuerecord2:
            res += 'pos {} {} {} {};'.format(self.glyphs1.asFea(), self.valuerecord1.asFea(), self.glyphs2.asFea(), self.valuerecord2.asFea())
        else:
            res += 'pos {} {} {};'.format(self.glyphs1.asFea(), self.glyphs2.asFea(), self.valuerecord1.asFea())
        return res