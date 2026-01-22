from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
def _get_A_ks(self, eq_params):
    non_precip_rids = self.eqsys.non_precip_rids(self.precipitates)
    return self.eqsys.stoichs_constants(self.eqsys.eq_constants(non_precip_rids, eq_params, self.small), self.rref_equil, backend=self.backend, non_precip_rids=non_precip_rids)