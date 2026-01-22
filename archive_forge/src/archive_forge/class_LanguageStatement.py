from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LanguageStatement(Statement):
    """A ``language`` statement within a feature."""

    def __init__(self, language, include_default=True, required=False, location=None):
        Statement.__init__(self, location)
        assert len(language) == 4
        self.language = language
        self.include_default = include_default
        self.required = required

    def build(self, builder):
        """Call the builder object's ``set_language`` callback."""
        builder.set_language(location=self.location, language=self.language, include_default=self.include_default, required=self.required)

    def asFea(self, indent=''):
        res = 'language {}'.format(self.language.strip())
        if not self.include_default:
            res += ' exclude_dflt'
        if self.required:
            res += ' required'
        res += ';'
        return res