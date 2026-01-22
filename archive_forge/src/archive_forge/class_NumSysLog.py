from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class NumSysLog(_NumSys):
    small = math.exp(-36)

    def pre_processor(self, x, params):
        return (np.log(np.asarray(x) + NumSysLog.small), params)

    def post_processor(self, x, params):
        return (np.exp(x), params)

    def internal_x0_cb(self, init_concs, params):
        return [0.1] * len(init_concs)

    def f(self, yvec, params):
        from pyneqsys.symbolic import linear_exprs
        init_concs, eq_params = self._inits_and_eq_params(params)
        A, ks = self._get_A_ks(eq_params)
        f_equil = mat_dot_vec(A, yvec, [-self.backend.log(k) for k in ks])
        B, comp_nrs = self.eqsys.composition_balance_vectors()
        f_preserv = linear_exprs(B, list(map(self.backend.exp, yvec)), mat_dot_vec(B, init_concs), rref=self.rref_preserv)
        return f_equil + f_preserv