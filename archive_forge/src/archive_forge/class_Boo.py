import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
class Boo(object):

    def __init__(self, a):
        pass

    def f(self, a):
        pass

    def g(self):
        pass
    foo = 'bar'

    class Bar(object):

        def a(self):
            pass