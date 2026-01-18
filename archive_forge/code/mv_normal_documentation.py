import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
return distribution of a full rank affine transform

        for full rank scale_matrix only

        Parameters
        ----------
        shift : array_like
            shift of mean
        scale_matrix : array_like
            linear transformation matrix

        Returns
        -------
        mvt : instance of MVT
            instance of multivariate t distribution given by affine
            transformation


        Notes
        -----

        This checks for eigvals<=0, so there are possible problems for cases
        with positive eigenvalues close to zero.

        see: http://www.statlect.com/mcdstu1.htm

        I'm not sure about general case, non-full rank transformation are not
        multivariate t distributed.

        y = a + B x

        where a is shift,
        B is full rank scale matrix with same dimension as sigma

        