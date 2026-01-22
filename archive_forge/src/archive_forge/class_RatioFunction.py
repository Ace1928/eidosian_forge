from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
class RatioFunction(AbstractRatioFunction):
    """Concrete implementation of a series of ratio functions."""
    ALLOWED_FUNCTIONS = dict(power2_decreasing_exp=['max', 'alpha'], smoothstep=['lower', 'upper'], smootherstep=['lower', 'upper'], inverse_smoothstep=['lower', 'upper'], inverse_smootherstep=['lower', 'upper'], power2_inverse_decreasing=['max'], power2_inverse_power2_decreasing=['max'])

    def power2_decreasing_exp(self, vals):
        """Get the evaluation of the ratio function f(x)=exp(-a*x)*(x-1)^2.

        The values (i.e. "x"), are scaled to the "max" parameter. The "a" constant
        correspond to the "alpha" parameter.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return power2_decreasing_exp(vals, edges=[0.0, self.__dict__['max']], alpha=self.__dict__['alpha'])

    def smootherstep(self, vals):
        """Get the evaluation of the smootherstep ratio function: f(x)=6*x^5-15*x^4+10*x^3.

        The values (i.e. "x"), are scaled between the "lower" and "upper" parameters.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return smootherstep(vals, edges=[self.__dict__['lower'], self.__dict__['upper']])

    def smoothstep(self, vals):
        """Get the evaluation of the smoothstep ratio function: f(x)=3*x^2-2*x^3.

        The values (i.e. "x"), are scaled between the "lower" and "upper" parameters.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return smoothstep(vals, edges=[self.__dict__['lower'], self.__dict__['upper']])

    def inverse_smootherstep(self, vals):
        """Get the evaluation of the "inverse" smootherstep ratio function: f(x)=1-(6*x^5-15*x^4+10*x^3).

        The values (i.e. "x"), are scaled between the "lower" and "upper" parameters.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return smootherstep(vals, edges=[self.__dict__['lower'], self.__dict__['upper']], inverse=True)

    def inverse_smoothstep(self, vals):
        """Get the evaluation of the "inverse" smoothstep ratio function: f(x)=1-(3*x^2-2*x^3).

        The values (i.e. "x"), are scaled between the "lower" and "upper" parameters.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return smoothstep(vals, edges=[self.__dict__['lower'], self.__dict__['upper']], inverse=True)

    def power2_inverse_decreasing(self, vals):
        """Get the evaluation of the ratio function f(x)=(x-1)^2 / x.

        The values (i.e. "x"), are scaled to the "max" parameter.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return power2_inverse_decreasing(vals, edges=[0.0, self.__dict__['max']])

    def power2_inverse_power2_decreasing(self, vals):
        """Get the evaluation of the ratio function f(x)=(x-1)^2 / x^2.

        The values (i.e. "x"), are scaled to the "max" parameter.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
        return power2_inverse_power2_decreasing(vals, edges=[0.0, self.__dict__['max']])