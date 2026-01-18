import os.path
from pyomo.common.fileutils import this_file_dir, import_file
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.opt import TerminationCondition
from io import StringIO
import logging
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_solver
import cyipopt as cyipopt_core
@unittest.skipIf(not pandas_available, 'Test uses pandas for data')
def test_parameter_estimation(self):
    data_fname = os.path.join(example_dir, 'external_grey_box', 'param_est', 'smalldata.csv')
    baseline = pandas.read_csv(data_fname)
    ex = import_file(os.path.join(example_dir, 'external_grey_box', 'param_est', 'generate_data.py'))
    df1 = ex.generate_data(5, 200, 5, 42)
    df2 = ex.generate_data_external(5, 200, 5, 42)
    pandas.testing.assert_frame_equal(df1, baseline, atol=0.001)
    pandas.testing.assert_frame_equal(df2, baseline, atol=0.001)
    ex = import_file(os.path.join(example_dir, 'external_grey_box', 'param_est', 'perform_estimation.py'))
    m = ex.perform_estimation_external(data_fname, solver_trace=False)
    self.assertAlmostEqual(pyo.value(m.UA), 204.43761, places=3)
    m = ex.perform_estimation_pyomo_only(data_fname, solver_trace=False)
    self.assertAlmostEqual(pyo.value(m.UA), 204.43761, places=3)