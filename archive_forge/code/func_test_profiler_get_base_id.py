import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def test_profiler_get_base_id(self):
    prof = profiler._Profiler('secret', base_id='1', parent_id='2')
    self.assertEqual(prof.get_base_id(), '1')