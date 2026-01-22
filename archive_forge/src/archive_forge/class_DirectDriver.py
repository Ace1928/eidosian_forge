import os
import json
import os.path
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL
import pyomo.neos
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
class DirectDriver(object):

    def _run(self, opt, constrained=True):
        m = _model(self.sense)
        with pyo.SolverManagerFactory('neos') as solver_manager:
            results = solver_manager.solve(m, opt=opt)
        expected_y = {(pyo.minimize, True): -1, (pyo.maximize, True): 1, (pyo.minimize, False): -10, (pyo.maximize, False): 10}[self.sense, constrained]
        self.assertEqual(results.solver[0].status, pyo.SolverStatus.ok)
        if constrained:
            self.assertAlmostEqual(pyo.value(m.x), 1, delta=1e-05)
        self.assertAlmostEqual(pyo.value(m.obj), expected_y, delta=1e-05)
        self.assertAlmostEqual(pyo.value(m.y), expected_y, delta=1e-05)