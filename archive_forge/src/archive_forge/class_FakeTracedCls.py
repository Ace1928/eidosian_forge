import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTracedCls(object):

    def method1(self, a, b, c=10):
        return a + b + c

    def method2(self, d, e):
        return d - e

    def method3(self, g=10, h=20):
        return g * h

    def _method(self, i):
        return i