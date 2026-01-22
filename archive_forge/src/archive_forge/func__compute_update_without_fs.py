import numpy as np
from ase.optimize.optimize import Dynamics
def _compute_update_without_fs(self, potentiostat_step_size, scale=1.0):
    """Only uses the forces to compute an orthogonal update vector"""
    self.step_size = self.maxstep * scale
    delta_s_perpendicular, delta_s_parallel, delta_s_drift = self.compute_step_contributions(potentiostat_step_size)
    dr_perpendicular = self.N * delta_s_perpendicular
    dr_parallel = delta_s_parallel * self.T
    D = self.create_drift_unit_vector(self.N, self.T)
    dr_drift = D * delta_s_drift
    dr = dr_parallel + dr_drift + dr_perpendicular
    dr = self.step_size * normalize(dr)
    return dr