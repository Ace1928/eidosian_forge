import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
class PlainMatrixArray:
    """Class that defines linalg operation on an array

    simplest version as benchmark

    linear algebra recipes for multivariate normal and linear
    regression calculations

    """

    def __init__(self, data=None, sym=None):
        if data is not None:
            if sym is None:
                self.x = np.asarray(data)
                self.m = np.dot(self.x.T, self.x)
            else:
                raise ValueError('data and sym cannot be both given')
        elif sym is not None:
            self.m = np.asarray(sym)
            self.x = np.eye(*self.m.shape)
        else:
            raise ValueError('either data or sym need to be given')

    @cache_readonly
    def minv(self):
        return np.linalg.inv(self.m)

    def m_y(self, y):
        return np.dot(self.m, y)

    def minv_y(self, y):
        return np.dot(self.minv, y)

    @cache_readonly
    def mpinv(self):
        return linalg.pinv(self.m)

    @cache_readonly
    def xpinv(self):
        return linalg.pinv(self.x)

    def yt_m_y(self, y):
        return np.dot(y.T, np.dot(self.m, y))

    def yt_minv_y(self, y):
        return np.dot(y.T, np.dot(self.minv, y))

    def y_m_yt(self, y):
        return np.dot(y, np.dot(self.m, y.T))

    def y_minv_yt(self, y):
        return np.dot(y, np.dot(self.minv, y.T))

    @cache_readonly
    def mdet(self):
        return linalg.det(self.m)

    @cache_readonly
    def mlogdet(self):
        return np.log(linalg.det(self.m))

    @cache_readonly
    def meigh(self):
        evals, evecs = linalg.eigh(self.m)
        sortind = np.argsort(evals)[::-1]
        return (evals[sortind], evecs[:, sortind])

    @cache_readonly
    def mhalf(self):
        evals, evecs = self.meigh
        return np.dot(np.diag(evals ** 0.5), evecs.T)

    @cache_readonly
    def minvhalf(self):
        evals, evecs = self.meigh
        return np.dot(evecs, 1.0 / np.sqrt(evals) * evecs.T)