from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
class DeltaCSMRatioFunction(AbstractRatioFunction):
    """
    Concrete implementation of a series of ratio functions applied to differences of
    continuous symmetry measures (DeltaCSM).

    Uses "finite" ratio functions.

    See the following reference for details:
    ChemEnv: a fast and robust coordination environment identification tool,
    D. Waroquiers et al., Acta Cryst. B 76, 683 (2020).
    """
    ALLOWED_FUNCTIONS = dict(smootherstep=['delta_csm_min', 'delta_csm_max'])

    def smootherstep(self, vals):
        """Get the evaluation of the smootherstep ratio function: f(x)=6*x^5-15*x^4+10*x^3.

        The DeltaCSM values (i.e. "x"), are scaled between the "delta_csm_min" and "delta_csm_max" parameters.

        Args:
            vals: DeltaCSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the DeltaCSM values.
        """
        return smootherstep(vals, edges=[self.__dict__['delta_csm_min'], self.__dict__['delta_csm_max']])