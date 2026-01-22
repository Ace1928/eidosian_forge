from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
class MisbehavingBytesSubtype(binary_type):

    def decode(self, encoding=None):
        return 'bad decode'

    def __str__(self):
        return 'bad __str__'

    def __bytes__(self):
        return b('bad __bytes__')