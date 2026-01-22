from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class MarkClassName(Expression):
    """A mark class name, such as ``@FRENCH_MARKS`` defined with ``markClass``.
    This must be instantiated with a :class:`MarkClass` object."""

    def __init__(self, markClass, location=None):
        Expression.__init__(self, location)
        assert isinstance(markClass, MarkClass)
        self.markClass = markClass

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return self.markClass.glyphSet()

    def asFea(self, indent=''):
        return '@' + self.markClass.name