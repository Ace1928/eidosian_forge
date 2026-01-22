from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def asFea(self, indent=''):
    res = indent + 'variation %s ' % self.name.strip()
    res += self.conditionset + ' '
    if self.use_extension:
        res += 'useExtension '
    res += '{\n'
    res += Block.asFea(self, indent=indent)
    res += indent + '} %s;\n' % self.name.strip()
    return res