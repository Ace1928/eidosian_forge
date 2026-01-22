from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
def _inits_and_eq_params(self, params):
    eq_params = params[self.eqsys.ns:]
    if not self.new_eq_params:
        assert not eq_params, 'Adjust number of parameters accordingly'
        eq_params = None
    return (params[:self.eqsys.ns], eq_params)