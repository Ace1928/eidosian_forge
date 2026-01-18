import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def test_stop_not_inited(self):
    profiler.clean()
    profiler.stop()