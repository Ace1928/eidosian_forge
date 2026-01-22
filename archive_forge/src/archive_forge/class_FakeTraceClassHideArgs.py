import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace_cls('a', info={'b': 20}, hide_args=True)
class FakeTraceClassHideArgs(FakeTracedCls):
    pass