import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
import gc
from io import StringIO
from itertools import zip_longest
import logging
import sys
import time
from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import (
from pyomo.environ import (
from pyomo.core.base.var import _VarData
def test_timer_depth_4_same_name(self):
    timer = self.make_timer_depth_4_same_name()
    root = timer.timers['root']
    root.flatten()
    self.assertAlmostEqual(root.total_time, 5.0)
    self.assertAlmostEqual(root.timers['a'].total_time, 1.0)