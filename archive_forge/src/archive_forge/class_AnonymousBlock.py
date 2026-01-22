from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class AnonymousBlock(Statement):
    """An anonymous data block."""

    def __init__(self, tag, content, location=None):
        Statement.__init__(self, location)
        self.tag = tag
        self.content = content

    def asFea(self, indent=''):
        res = 'anon {} {{\n'.format(self.tag)
        res += self.content
        res += '}} {};\n\n'.format(self.tag)
        return res