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
def test_report_timing(self):
    ref = '\n           (0(\\.\\d+)?) seconds to construct Block ConcreteModel; 1 index total\n           (0(\\.\\d+)?) seconds to construct RangeSet FiniteScalarRangeSet; 1 index total\n           (0(\\.\\d+)?) seconds to construct Var x; 2 indices total\n           (0(\\.\\d+)?) seconds to construct Var y; 0 indices total\n           (0(\\.\\d+)?) seconds to construct Suffix Suffix\n           (0(\\.\\d+)?) seconds to apply Transformation RelaxIntegerVars \\(in-place\\)\n           '.strip()
    xfrm = TransformationFactory('core.relax_integer_vars')
    try:
        with capture_output() as out:
            report_timing()
            m = ConcreteModel()
            m.r = RangeSet(2)
            m.x = Var(m.r)
            m.y = Var(Any, dense=False)
            xfrm.apply_to(m)
        result = out.getvalue().strip()
        self.maxDiff = None
        for l, r in zip(result.splitlines(), ref.splitlines()):
            self.assertRegex(str(l.strip()), str(r.strip()))
    finally:
        report_timing(False)
    os = StringIO()
    try:
        report_timing(os)
        m = ConcreteModel()
        m.r = RangeSet(2)
        m.x = Var(m.r)
        m.y = Var(Any, dense=False)
        xfrm.apply_to(m)
        result = os.getvalue().strip()
        self.maxDiff = None
        for l, r in zip(result.splitlines(), ref.splitlines()):
            self.assertRegex(str(l.strip()), str(r.strip()))
    finally:
        report_timing(False)
    buf = StringIO()
    with LoggingIntercept(buf, 'pyomo'):
        m = ConcreteModel()
        m.r = RangeSet(2)
        m.x = Var(m.r)
        m.y = Var(Any, dense=False)
        xfrm.apply_to(m)
        result = os.getvalue().strip()
        self.maxDiff = None
        for l, r in zip(result.splitlines(), ref.splitlines()):
            self.assertRegex(str(l.strip()), str(r.strip()))
        self.assertEqual(buf.getvalue().strip(), '')