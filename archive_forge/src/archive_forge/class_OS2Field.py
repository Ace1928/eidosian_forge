from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class OS2Field(Statement):
    """An entry in the ``OS/2`` table. Most ``values`` should be numbers or
    strings, apart from when the key is ``UnicodeRange``, ``CodePageRange``
    or ``Panose``, in which case it should be an array of integers."""

    def __init__(self, key, value, location=None):
        Statement.__init__(self, location)
        self.key = key
        self.value = value

    def build(self, builder):
        """Calls the builder object's ``add_os2_field`` callback."""
        builder.add_os2_field(self.key, self.value)

    def asFea(self, indent=''):

        def intarr2str(x):
            return ' '.join(map(str, x))
        numbers = ('FSType', 'TypoAscender', 'TypoDescender', 'TypoLineGap', 'winAscent', 'winDescent', 'XHeight', 'CapHeight', 'WeightClass', 'WidthClass', 'LowerOpSize', 'UpperOpSize')
        ranges = ('UnicodeRange', 'CodePageRange')
        keywords = dict([(x.lower(), [x, str]) for x in numbers])
        keywords.update([(x.lower(), [x, intarr2str]) for x in ranges])
        keywords['panose'] = ['Panose', intarr2str]
        keywords['vendor'] = ['Vendor', lambda y: '"{}"'.format(y)]
        if self.key in keywords:
            return '{} {};'.format(keywords[self.key][0], keywords[self.key][1](self.value))
        return ''