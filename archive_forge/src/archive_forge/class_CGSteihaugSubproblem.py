import math
import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""

    def solve(self, trust_radius):
        """
        Solve the subproblem using a conjugate gradient method.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        This is algorithm (7.2) of Nocedal and Wright 2nd edition.
        Only the function that computes the Hessian-vector product is required.
        The Hessian itself is not required, and the Hessian does
        not need to be positive semidefinite.
        """
        p_origin = np.zeros_like(self.jac)
        tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag
        if self.jac_mag < tolerance:
            hits_boundary = False
            return (p_origin, hits_boundary)
        z = p_origin
        r = self.jac
        d = -r
        while True:
            Bd = self.hessp(d)
            dBd = np.dot(d, Bd)
            if dBd <= 0:
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return (p_boundary, hits_boundary)
            r_squared = np.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if scipy.linalg.norm(z_next) >= trust_radius:
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return (p_boundary, hits_boundary)
            r_next = r + alpha * Bd
            r_next_squared = np.dot(r_next, r_next)
            if math.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return (z_next, hits_boundary)
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d
            z = z_next
            r = r_next
            d = d_next