import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
class MVNormal(MVElliptical):
    """Class for Multivariate Normal Distribution

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    """
    __name__ == 'Multivariate Normal Distribution'

    def rvs(self, size=1):
        """random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        """
        return np.random.multivariate_normal(self.mean, self.sigma, size=size)

    def logpdf(self, x):
        """logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        does not work now because of dot in whiten

        """
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened ** 2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2.0 * np.pi)
        llf -= self.logdetsigma
        llf *= 0.5
        return llf

    def cdf(self, x, **kwds):
        """cumulative distribution function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector
        kwds : dict
            contains options for the numerical calculation of the cdf

        Returns
        -------
        cdf : float or array
            probability density value of each random vector

        """
        return mvnormcdf(x, self.mean, self.cov, **kwds)

    @property
    def cov(self):
        """covariance matrix"""
        return self.sigma

    def affine_transformed(self, shift, scale_matrix):
        """return distribution of an affine transform

        for full rank scale_matrix only

        Parameters
        ----------
        shift : array_like
            shift of mean
        scale_matrix : array_like
            linear transformation matrix

        Returns
        -------
        mvt : instance of MVNormal
            instance of multivariate normal distribution given by affine
            transformation

        Notes
        -----
        the affine transformation is defined by
        y = a + B x

        where a is shift,
        B is a scale matrix for the linear transformation

        Notes
        -----
        This should also work to select marginal distributions, but not
        tested for this case yet.

        currently only tested because it's called by standardized

        """
        B = scale_matrix
        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVNormal(mean_new, sigma_new)

    def conditional(self, indices, values):
        """return conditional distribution

        indices are the variables to keep, the complement is the conditioning
        set
        values are the values of the conditioning variables

        \\bar{\\mu} = \\mu_1 + \\Sigma_{12} \\Sigma_{22}^{-1} \\left( a - \\mu_2 \\right)

        and covariance matrix

        \\overline{\\Sigma} = \\Sigma_{11} - \\Sigma_{12} \\Sigma_{22}^{-1} \\Sigma_{21}.T

        Parameters
        ----------
        indices : array_like, int
            list of indices of variables in the marginal distribution
        given : array_like
            values of the conditioning variables

        Returns
        -------
        mvn : instance of MVNormal
            new instance of the MVNormal class that contains the conditional
            distribution of the variables given in indices for given
             values of the excluded variables.


        """
        keep = np.asarray(indices)
        given = np.asarray([i for i in range(self.nvars) if i not in keep])
        sigmakk = self.sigma[keep[:, None], keep]
        sigmagg = self.sigma[given[:, None], given]
        sigmakg = self.sigma[keep[:, None], given]
        sigmagk = self.sigma[given[:, None], keep]
        sigma_new = sigmakk - np.dot(sigmakg, np.linalg.solve(sigmagg, sigmagk))
        mean_new = self.mean[keep] + np.dot(sigmakg, np.linalg.solve(sigmagg, values - self.mean[given]))
        return MVNormal(mean_new, sigma_new)