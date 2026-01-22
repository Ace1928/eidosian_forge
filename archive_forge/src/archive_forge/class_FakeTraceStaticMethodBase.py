import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTraceStaticMethodBase(FakeTracedCls):

    @staticmethod
    def static_method(arg):
        return arg