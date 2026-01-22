from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
class DuckValue(object):

    def __init__(self, *args):
        self.value = Value(*args)

    def _asdict(self):
        return self.value._asdict()