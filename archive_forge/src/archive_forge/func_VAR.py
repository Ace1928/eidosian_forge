import numpy as np
from scipy import signal
def VAR(x, B, const=0):
    """ multivariate linear filter

    Parameters
    ----------
    x: (TxK) array
        columns are variables, rows are observations for time period
    B: (PxKxK) array
        b_t-1 is bottom "row", b_t-P is top "row" when printing
        B(:,:,0) is lag polynomial matrix for variable 1
        B(:,:,k) is lag polynomial matrix for variable k
        B(p,:,k) is pth lag for variable k
        B[p,:,:].T corresponds to A_p in Wikipedia
    const : float or array (not tested)
        constant added to autoregression

    Returns
    -------
    xhat: (TxK) array
        filtered, predicted values of x array

    Notes
    -----
    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) }  for all i = 0,K-1, for all t=p..T

    xhat does not include the forecasting observation, xhat(T+1),
    xhat is 1 row shorter than signal.correlate

    References
    ----------
    https://en.wikipedia.org/wiki/Vector_Autoregression
    https://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)
    """
    p = B.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    for t in range(p, T):
        xhat[t, :] = const + (x[t - p:t, :, np.newaxis] * B).sum(axis=1).sum(axis=0)
    return xhat