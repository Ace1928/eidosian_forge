from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.symbol import Dummy
from sympy.integrals.integrals import Integral
from sympy.stats.rv import (NamedArgsMixin, random_symbols, _symbol_converter,
from sympy.stats.crv import ContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import DiscreteDistribution, SingleDiscretePSpace
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.crv_types import ContinuousDistributionHandmade
from sympy.stats.drv_types import DiscreteDistributionHandmade
from sympy.stats.frv_types import FiniteDistributionHandmade
def _marginalise(self, expr, rv, evaluate):
    if isinstance(rv.pspace.distribution, SingleFiniteDistribution):
        rv_dens = rv.pspace.distribution.pmf(rv)
    else:
        rv_dens = rv.pspace.distribution.pdf(rv)
    rv_dom = rv.pspace.domain.set
    if rv.pspace.is_Discrete or rv.pspace.is_Finite:
        expr = Sum(expr * rv_dens, (rv, rv_dom._inf, rv_dom._sup))
    else:
        expr = Integral(expr * rv_dens, (rv, rv_dom._inf, rv_dom._sup))
    if evaluate:
        return expr.doit()
    return expr