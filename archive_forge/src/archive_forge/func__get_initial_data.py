import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data, run_cstr_mpc
def _get_initial_data(self):
    initial_data = mpc.ScalarData({'flow_in[*]': 0.3})
    return get_steady_state_data(initial_data)