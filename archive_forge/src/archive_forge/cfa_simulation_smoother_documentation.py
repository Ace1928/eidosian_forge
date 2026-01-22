import numpy as np
from . import tools

        Perform simulation smoothing (via Cholesky factor algorithm)

        Does not return anything, but populates the object's `simulated_state`
        attribute, and also makes available the attributes `posterior_mean`,
        `posterior_cov`, and `posterior_cov_inv_chol_sparse`.

        Parameters
        ----------
        variates : array_like, optional
            Random variates, distributed standard Normal. Usually only
            specified if results are to be replicated (e.g. to enforce a seed)
            or for testing. If not specified, random variates are drawn. Must
            be shaped (nobs, k_states).

        Notes
        -----
        The first step in simulating from the joint posterior of the state
        vector conditional on the data is to compute the two relevant moments
        of the joint posterior distribution:

        .. math::

            \alpha \mid Y_n \sim N(\hat \alpha, Var(\alpha \mid Y_n))

        Let :math:`L L' = Var(\alpha \mid Y_n)^{-1}`. Then simulation proceeds
        according to the following steps:

        1. Draw :math:`u \sim N(0, I)`
        2. Compute :math:`x = \hat \alpha + (L')^{-1} u`

        And then :math:`x` is a draw from the joint posterior of the states.
        The output of the function is as follows:

        - The simulated draw :math:`x` is held in the `simulated_state`
          attribute.
        - The posterior mean :math:`\hat \alpha` is held in the
          `posterior_mean` attribute.
        - The (lower triangular) Cholesky factor of the inverse posterior
          covariance matrix, :math:`L`, is held in sparse diagonal banded
          storage in the `posterior_cov_inv_chol` attribute.
        - The posterior covariance matrix :math:`Var(\alpha \mid Y_n)` can be
          computed on demand by accessing the `posterior_cov` property. Note
          that this matrix can be extremely large, so care must be taken when
          accessing this property. In most cases, it will be preferred to make
          use of the `posterior_cov_inv_chol` attribute rather than the
          `posterior_cov` attribute.

        