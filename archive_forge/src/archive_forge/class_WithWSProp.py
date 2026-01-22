import re
import unittest
from wsme import exc
from wsme import types
class WithWSProp(object):

    def __init__(self):
        self._aint = 0

    def get_aint(self):
        return self._aint

    def set_aint(self, value):
        self._aint = value
    aint = types.wsproperty(int, get_aint, set_aint, mandatory=True)