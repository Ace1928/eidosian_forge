import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
class QIFCovariance:
    """
    A covariance model for quadratic inference function regression.

    The mat method returns a basis matrix B such that the inverse
    of the working covariance lies in the linear span of the
    basis matrices.

    Subclasses should set the number of basis matrices `num_terms`,
    so that `mat(d, j)` for j=0, ..., num_terms-1 gives the basis
    of dimension d.`
    """

    def mat(self, dim, term):
        """
        Returns the term'th basis matrix, which is a dim x dim
        matrix.
        """
        raise NotImplementedError