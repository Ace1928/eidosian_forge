from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class FontRevisionStatement(Statement):
    """A ``head`` table ``FontRevision`` statement. ``revision`` should be a
    number, and will be formatted to three significant decimal places."""

    def __init__(self, revision, location=None):
        Statement.__init__(self, location)
        self.revision = revision

    def build(self, builder):
        builder.set_font_revision(self.location, self.revision)

    def asFea(self, indent=''):
        return 'FontRevision {:.3f};'.format(self.revision)