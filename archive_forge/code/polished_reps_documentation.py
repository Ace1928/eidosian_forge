from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *

        Get the polished cusp shape for this representation::

          sage: M = ManifoldHP('m015')
          sage: rho = M.polished_holonomy(bits_prec=100)
          sage: rho.cusp_shape()   # doctest: +NUMERIC24
          -0.49024466750661447990098220731 + 2.9794470664789769463726817144*I

        