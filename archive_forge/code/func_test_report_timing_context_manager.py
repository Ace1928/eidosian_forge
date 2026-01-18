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
def test_report_timing_context_manager(self):
    ref = '\n           (0(\\.\\d+)?) seconds to construct Var x; 2 indices total\n           (0(\\.\\d+)?) seconds to construct Var y; 0 indices total\n           (0(\\.\\d+)?) seconds to construct Suffix Suffix\n           (0(\\.\\d+)?) seconds to apply Transformation RelaxIntegerVars \\(in-place\\)\n           '.strip()
    xfrm = TransformationFactory('core.relax_integer_vars')
    model = AbstractModel()
    model.r = RangeSet(2)
    model.x = Var(model.r)
    model.y = Var(Any, dense=False)
    OS = StringIO()
    with report_timing(False):
        with report_timing(OS):
            with report_timing(False):
                with capture_output() as OUT:
                    m = model.create_instance()
                    xfrm.apply_to(m)
                self.assertEqual(OUT.getvalue(), '')
                self.assertEqual(OS.getvalue(), '')
            with capture_output() as OUT:
                m = model.create_instance()
                xfrm.apply_to(m)
            self.assertEqual(OUT.getvalue(), '')
            result = OS.getvalue().strip()
            self.maxDiff = None
            for l, r in zip_longest(result.splitlines(), ref.splitlines()):
                self.assertRegex(str(l.strip()), str(r.strip()))
        with capture_output() as OUT:
            m = model.create_instance()
            xfrm.apply_to(m)
        self.assertEqual(OUT.getvalue(), '')
        self.assertEqual(result, OS.getvalue().strip())