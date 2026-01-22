import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTraceClassMethodBase(FakeTracedCls):

    @classmethod
    def class_method(cls, arg):
        return arg