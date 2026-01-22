import numpy as np
from statsmodels.duration.hazard_regression import PHReg
def _kernel_cumincidence(time, status, exog, kfunc, freq_weights, dimred=True):
    """
    Calculates cumulative incidence functions using kernels.

    Parameters
    ----------
    time : array_like
        The observed time values
    status : array_like
        The status values.  status == 0 indicates censoring,
        status == 1, 2, ... are the events.
    exog : array_like
        Covariates such that censoring becomes independent of
        outcome times conditioned on the covariate values.
    kfunc : function
        A kernel function
    freq_weights : array_like
        Optional frequency weights
    dimred : bool
        If True, proportional hazards regression models are used to
        reduce exog to two columns by predicting overall events and
        censoring in two separate models.  If False, exog is used
        directly for calculating kernel weights without dimension
        reduction.
    """
    ii = np.argsort(time)
    time = time[ii]
    status = status[ii]
    exog = exog[ii, :]
    nobs = len(time)
    utime, rtime = np.unique(time, return_inverse=True)
    ie = np.searchsorted(time, utime, side='right') - 1
    ngrp = int(status.max())
    statusa = (status >= 1).astype(np.float64)
    if freq_weights is not None:
        freq_weights = freq_weights / freq_weights.sum()
    ip = []
    sp = [None] * nobs
    n_risk = [None] * nobs
    kd = [None] * nobs
    for k in range(ngrp):
        status0 = (status == k + 1).astype(np.float64)
        if dimred:
            sfe = PHReg(time, exog, status0).fit()
            fitval_e = sfe.predict().predicted_values
            sfc = PHReg(time, exog, 1 - status0).fit()
            fitval_c = sfc.predict().predicted_values
            exog2d = np.hstack((fitval_e[:, None], fitval_c[:, None]))
            exog2d -= exog2d.mean(0)
            exog2d /= exog2d.std(0)
        else:
            exog2d = exog
        ip0 = 0
        for i in range(nobs):
            if k == 0:
                kd1 = exog2d - exog2d[i, :]
                kd1 = kfunc(kd1)
                kd[i] = kd1
            if k == 0:
                denom = np.cumsum(kd[i][::-1])[::-1]
                num = kd[i] * statusa
                rat = num / denom
                tr = 1e-15
                ii = np.flatnonzero((denom < tr) & (num < tr))
                rat[ii] = 0
                ratc = 1 - rat
                ratc = np.clip(ratc, 1e-10, np.inf)
                lrat = np.log(ratc)
                prat = np.cumsum(lrat)[ie]
                sf = np.exp(prat)
                sp[i] = np.r_[1, sf[:-1]]
                n_risk[i] = denom[ie]
            d0 = np.bincount(rtime, weights=status0 * kd[i], minlength=len(utime))
            ip1 = np.cumsum(sp[i] * d0 / n_risk[i])
            jj = len(ip1) - np.searchsorted(n_risk[i][::-1], 1)
            if jj < len(ip1):
                ip1[jj:] = ip1[jj - 1]
            if freq_weights is None:
                ip0 += ip1
            else:
                ip0 += freq_weights[i] * ip1
        if freq_weights is None:
            ip0 /= nobs
        ip.append(ip0)
    return (utime, ip)