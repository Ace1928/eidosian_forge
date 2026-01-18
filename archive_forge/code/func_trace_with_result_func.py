import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace('hide_result', hide_result=False)
def trace_with_result_func(a, i=10):
    return (a, i)