import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def varinversefilter(ar, nobs, version=1):
    """creates inverse ar filter (MA representation) recursively

    The VAR lag polynomial is defined by ::

        ar(L) y_t = u_t  or
        y_t = -ar_{-1}(L) y_{t-1} + u_t

    the returned lagpolynomial is arinv(L)=ar^{-1}(L) in ::

        y_t = arinv(L) u_t



    Parameters
    ----------
    ar : ndarray, (nlags,nvars,nvars)
        matrix lagpolynomial, currently no exog
        first row should be identity

    Returns
    -------
    arinv : ndarray, (nobs,nvars,nvars)


    Notes
    -----

    """
    nlags, nvars, nvarsex = ar.shape
    if nvars != nvarsex:
        print('exogenous variables not implemented not tested')
    arinv = np.zeros((nobs + 1, nvarsex, nvars))
    arinv[0, :, :] = ar[0]
    arinv[1:nlags, :, :] = -ar[1:]
    if version == 1:
        for i in range(2, nobs + 1):
            tmp = np.zeros((nvars, nvars))
            for p in range(1, nlags):
                tmp += np.dot(-ar[p], arinv[i - p, :, :])
            arinv[i, :, :] = tmp
    if version == 0:
        for i in range(nlags + 1, nobs + 1):
            print(ar[1:].shape, arinv[i - 1:i - nlags:-1, :, :].shape)
            raise NotImplementedError('waiting for generalized ufuncs or something')
    return arinv