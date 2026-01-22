import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
class RaiserClass(object):
    raiser = Raiser()

    @staticmethod
    def existing(a, b):
        return a + b