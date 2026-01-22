from statsmodels.compat.python import lzip
import numpy as np
create new results instance with robust covariance as default

    Parameters
    ----------
    cov_type : str
        the type of robust sandwich estimator to use. see Notes below
    use_t : bool
        If true, then the t distribution is used for inference.
        If false, then the normal distribution is used.
    kwds : depends on cov_type
        Required or optional arguments for robust covariance calculation.
        see Notes below

    Returns
    -------
    results : results instance
        This method creates a new results instance with the requested
        robust covariance as the default covariance of the parameters.
        Inferential statistics like p-values and hypothesis tests will be
        based on this covariance matrix.

    Notes
    -----
    Warning: Some of the options and defaults in cov_kwds may be changed in a
    future version.

    The covariance keywords provide an option 'scaling_factor' to adjust the
    scaling of the covariance matrix, that is the covariance is multiplied by
    this factor if it is given and is not `None`. This allows the user to
    adjust the scaling of the covariance matrix to match other statistical
    packages.
    For example, `scaling_factor=(nobs - 1.) / (nobs - k_params)` provides a
    correction so that the robust covariance matrices match those of Stata in
    some models like GLM and discrete Models.

    The following covariance types and required or optional arguments are
    currently available:

    - 'HC0', 'HC1', 'HC2', 'HC3': heteroscedasticity robust covariance

      - no keyword arguments

    - 'HAC': heteroskedasticity-autocorrelation robust covariance

      ``maxlags`` :  integer, required
        number of lags to use

      ``kernel`` : {callable, str}, optional
        kernels currently available kernels are ['bartlett', 'uniform'],
        default is Bartlett

      ``use_correction``: bool, optional
        If true, use small sample correction

    - 'cluster': clustered covariance estimator

      ``groups`` : array_like[int], required :
        Integer-valued index of clusters or groups.

      ``use_correction``: bool, optional
        If True the sandwich covariance is calculated with a small
        sample correction.
        If False the sandwich covariance is calculated without
        small sample correction.

      ``df_correction``: bool, optional
        If True (default), then the degrees of freedom for the
        inferential statistics and hypothesis tests, such as
        pvalues, f_pvalue, conf_int, and t_test and f_test, are
        based on the number of groups minus one instead of the
        total number of observations minus the number of explanatory
        variables. `df_resid` of the results instance is also
        adjusted. When `use_t` is also True, then pvalues are
        computed using the Student's t distribution using the
        corrected values. These may differ substantially from
        p-values based on the normal is the number of groups is
        small.
        If False, then `df_resid` of the results instance is not
        adjusted.


    - 'hac-groupsum': Driscoll and Kraay, heteroscedasticity and
      autocorrelation robust covariance for panel data
      # TODO: more options needed here

      ``time`` : array_like, required
        index of time periods
      ``maxlags`` : integer, required
        number of lags to use
      ``kernel`` : {callable, str}, optional
        The available kernels are ['bartlett', 'uniform']. The default is
        Bartlett.
      ``use_correction`` : {False, 'hac', 'cluster'}, optional
        If False the the sandwich covariance is calculated without small
        sample correction. If `use_correction = 'cluster'` (default),
        then the same small sample correction as in the case of
        `covtype='cluster'` is used.
      ``df_correction`` : bool, optional
        The adjustment to df_resid, see cov_type 'cluster' above

    - 'hac-panel': heteroscedasticity and autocorrelation robust standard
      errors in panel data. The data needs to be sorted in this case, the
      time series for each panel unit or cluster need to be stacked. The
      membership to a time series of an individual or group can be either
      specified by group indicators or by increasing time periods. One of
      ``groups`` or ``time`` is required. # TODO: we need more options here

      ``groups`` : array_like[int]
        indicator for groups
      ``time`` : array_like[int]
        index of time periods
      ``maxlags`` : int, required
        number of lags to use
      ``kernel`` : {callable, str}, optional
        Available kernels are ['bartlett', 'uniform'], default
        is Bartlett
      ``use_correction`` : {False, 'hac', 'cluster'}, optional
        If False the sandwich covariance is calculated without
        small sample correction.
      ``df_correction`` : bool, optional
        Adjustment to df_resid, see cov_type 'cluster' above

    **Reminder**: ``use_correction`` in "hac-groupsum" and "hac-panel" is
    not bool, needs to be in {False, 'hac', 'cluster'}.

    .. todo:: Currently there is no check for extra or misspelled keywords,
         except in the case of cov_type `HCx`
    