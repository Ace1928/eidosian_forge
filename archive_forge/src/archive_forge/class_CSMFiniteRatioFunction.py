from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
class CSMFiniteRatioFunction(AbstractRatioFunction):
    """Concrete implementation of a series of ratio functions applied to the continuous symmetry measure (CSM).

    Uses "finite" ratio functions.

    See the following reference for details:
    ChemEnv: a fast and robust coordination environment identification tool,
    D. Waroquiers et al., Acta Cryst. B 76, 683 (2020).
    """
    ALLOWED_FUNCTIONS = dict(power2_decreasing_exp=['max_csm', 'alpha'], smoothstep=['lower_csm', 'upper_csm'], smootherstep=['lower_csm', 'upper_csm'])

    def power2_decreasing_exp(self, vals):
        """Get the evaluation of the ratio function f(x)=exp(-a*x)*(x-1)^2.

        The CSM values (i.e. "x"), are scaled to the "max_csm" parameter. The "a" constant
        correspond to the "alpha" parameter.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
        return power2_decreasing_exp(vals, edges=[0.0, self.__dict__['max_csm']], alpha=self.__dict__['alpha'])

    def smootherstep(self, vals):
        """Get the evaluation of the smootherstep ratio function: f(x)=6*x^5-15*x^4+10*x^3.

        The CSM values (i.e. "x"), are scaled between the "lower_csm" and "upper_csm" parameters.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
        return smootherstep(vals, edges=[self.__dict__['lower_csm'], self.__dict__['upper_csm']], inverse=True)

    def smoothstep(self, vals):
        """Get the evaluation of the smoothstep ratio function: f(x)=3*x^2-2*x^3.

        The CSM values (i.e. "x"), are scaled between the "lower_csm" and "upper_csm" parameters.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
        return smootherstep(vals, edges=[self.__dict__['lower_csm'], self.__dict__['upper_csm']], inverse=True)

    def fractions(self, data):
        """Get the fractions from the CSM ratio function applied to the data.

        Args:
            data: List of CSM values to estimate fractions.

        Returns:
            Corresponding fractions for each CSM.
        """
        if len(data) == 0:
            return None
        total = np.sum([self.eval(dd) for dd in data])
        if total > 0.0:
            return [self.eval(dd) / total for dd in data]
        return None

    def mean_estimator(self, data):
        """Get the weighted CSM using this CSM ratio function applied to the data.

        Args:
            data: List of CSM values to estimate the weighted CSM.

        Returns:
            Weighted CSM from this ratio function.
        """
        if len(data) == 0:
            return None
        if len(data) == 1:
            return data[0]
        fractions = self.fractions(data)
        if fractions is None:
            return None
        return np.sum(np.array(fractions) * np.array(data))
    ratios = fractions