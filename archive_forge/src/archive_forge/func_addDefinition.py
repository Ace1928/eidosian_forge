from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def addDefinition(self, definition):
    """Add a :class:`MarkClassDefinition` statement to this mark class."""
    assert isinstance(definition, MarkClassDefinition)
    self.definitions.append(definition)
    for glyph in definition.glyphSet():
        if glyph in self.glyphs:
            otherLoc = self.glyphs[glyph].location
            if otherLoc is None:
                end = ''
            else:
                end = f' at {otherLoc}'
            raise FeatureLibError('Glyph %s already defined%s' % (glyph, end), definition.location)
        self.glyphs[glyph] = definition