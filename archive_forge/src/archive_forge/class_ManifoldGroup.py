from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
class ManifoldGroup(MatrixRepresentation):

    def __init__(self, gens, relators, peripheral_curves=None, matrices=None):
        MatrixRepresentation.__init__(self, gens, relators, matrices)
        self._peripheral_curves = peripheral_curves

    def peripheral_curves(self):
        return self._peripheral_curves

    def SL2C(self, word):
        return self(word)

    def check_representation(self):
        relator_matrices = (self.SL2C(R) for R in self.relators())
        return max((projective_distance(A, identity(A)) for A in relator_matrices))

    def cusp_shape(self, cusp_num=0):
        """
        Get the polished cusp shape for this representation::

          sage: M = ManifoldHP('m015')
          sage: rho = M.polished_holonomy(bits_prec=100)
          sage: rho.cusp_shape()   # doctest: +NUMERIC24
          -0.49024466750661447990098220731 + 2.9794470664789769463726817144*I

        """
        M, L = [self.SL2C(w) for w in self.peripheral_curves()[cusp_num]]
        C = extend_to_basis(parabolic_eigenvector(M))
        M, L = [make_trace_2(C ** (-1) * A * C) for A in [M, L]]
        z = L[0][1] / M[0][1]
        return z.conjugate()

    def lift_to_SL2C(self):
        MatrixRepresentation.lift_to_SL2C(self)
        phi = MapToFreeAbelianization(self)
        meridian = self.peripheral_curves()[0][0]
        meridian_trace = self(meridian).trace()
        if phi.rank == 1 and phi(meridian) % 2 != 0 and (meridian_trace < 0):

            def twist(g, gen_image):
                return gen_image if phi(g)[0] % 2 == 0 else -gen_image
            self._matrices = [twist(g, M) for g, M in zip(self._gens, self._matrices)]
            self._build_hom_dict()
            assert self.is_nonprojective_representation()
            assert self(meridian).trace() > 0

    def all_lifts_to_SL2C(self):
        ans = []
        self.lift_to_SL2C()
        base_gen_images = [self(g) for g in self.generators()]
        pos_signs = product(*[(1, -1)] * len(base_gen_images))
        for signs in pos_signs:
            beta = ManifoldGroup(self.generators(), self.relators(), self.peripheral_curves(), [s * A for s, A in zip(signs, base_gen_images)])
            if beta.is_nonprojective_representation():
                ans.append(beta)
        return ans

    def __repr__(self):
        return 'Generators:\n   %s\nRelators:\n   %s' % (','.join(self.generators()), '\n   '.join(self.relators()))