from unittest import TestCase
import simplejson
from simplejson.compat import text_type
class WonkyTextSubclass(text_type):

    def __getslice__(self, start, end):
        return self.__class__('not what you wanted!')