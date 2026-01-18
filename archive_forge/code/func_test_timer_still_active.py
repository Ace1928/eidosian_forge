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
def test_timer_still_active(self):
    timer = HierarchicalTimer()
    timer.start('a')
    timer.stop('a')
    timer.start('b')
    msg = 'Cannot flatten.*while any timers are active'
    with self.assertRaisesRegex(RuntimeError, msg):
        timer.flatten()
    timer.stop('b')