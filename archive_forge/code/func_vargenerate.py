import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def vargenerate(ar, u, initvalues=None):
    """generate an VAR process with errors u

    similar to gauss
    uses loop

    Parameters
    ----------
    ar : array (nlags,nvars,nvars)
        matrix lagpolynomial
    u : array (nobs,nvars)
        exogenous variable, error term for VAR

    Returns
    -------
    sar : array (1+nobs,nvars)
        sample of var process, inverse filtered u
        does not trim initial condition y_0 = 0

    Examples
    --------
    # generate random sample of VAR
    nobs, nvars = 10, 2
    u = numpy.random.randn(nobs,nvars)
    a21 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.8,  0. ],
                     [ 0.,  -0.6]]])
    vargenerate(a21,u)

    # Impulse Response to an initial shock to the first variable
    imp = np.zeros((nobs, nvars))
    imp[0,0] = 1
    vargenerate(a21,imp)

    """
    nlags, nvars, nvarsex = ar.shape
    nlagsm1 = nlags - 1
    nobs = u.shape[0]
    if nvars != nvarsex:
        print('exogenous variables not implemented not tested')
    if u.shape[1] != nvars:
        raise ValueError('u needs to have nvars columns')
    if initvalues is None:
        sar = np.zeros((nobs + nlagsm1, nvars))
        start = nlagsm1
    else:
        start = max(nlagsm1, initvalues.shape[0])
        sar = np.zeros((nobs + start, nvars))
        sar[start - initvalues.shape[0]:start] = initvalues
    sar[start:] = u
    for i in range(start, start + nobs):
        for p in range(1, nlags):
            sar[i] += np.dot(sar[i - p, :], -ar[p])
    return sar