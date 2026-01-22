from pyomo.core.base import Transformation, TransformationFactory
import pyomo.gdp.plugins.multiple_bigm

    Converts a model containing PiecewiseLinearFunctions to a an equivalent
    MIP via the Convex Combination method from [1]. Note that,
    while this model probably resolves to the model described in [1] after
    presolve, the Pyomo version is not as simplified.

    References
    ----------
    [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
        for nonseparable piecewise-linear optimization: unifying framework
        and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
        2010.
    