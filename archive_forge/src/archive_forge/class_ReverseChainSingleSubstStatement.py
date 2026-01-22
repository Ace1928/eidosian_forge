from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ReverseChainSingleSubstStatement(Statement):
    """A reverse chaining substitution statement. You don't see those every day.

    Note the unusual argument order: ``suffix`` comes `before` ``glyphs``.
    ``old_prefix``, ``old_suffix``, ``glyphs`` and ``replacements`` should be
    lists of `glyph-containing objects`_. ``glyphs`` and ``replacements`` should
    be one-item lists.
    """

    def __init__(self, old_prefix, old_suffix, glyphs, replacements, location=None):
        Statement.__init__(self, location)
        self.old_prefix, self.old_suffix = (old_prefix, old_suffix)
        self.glyphs = glyphs
        self.replacements = replacements

    def build(self, builder):
        prefix = [p.glyphSet() for p in self.old_prefix]
        suffix = [s.glyphSet() for s in self.old_suffix]
        originals = self.glyphs[0].glyphSet()
        replaces = self.replacements[0].glyphSet()
        if len(replaces) == 1:
            replaces = replaces * len(originals)
        builder.add_reverse_chain_single_subst(self.location, prefix, suffix, dict(zip(originals, replaces)))

    def asFea(self, indent=''):
        res = 'rsub '
        if len(self.old_prefix) or len(self.old_suffix):
            if len(self.old_prefix):
                res += ' '.join((asFea(g) for g in self.old_prefix)) + ' '
            res += ' '.join((asFea(g) + "'" for g in self.glyphs))
            if len(self.old_suffix):
                res += ' ' + ' '.join((asFea(g) for g in self.old_suffix))
        else:
            res += ' '.join(map(asFea, self.glyphs))
        res += ' by {};'.format(' '.join((asFea(g) for g in self.replacements)))
        return res