import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def test_scale_oneway(data, method='bf', center='median', transform='abs', trim_frac_mean=0.1, trim_frac_anova=0.0):
    """Oneway Anova test for equal scale, variance or dispersion

    This hypothesis test performs a oneway anova test on transformed data and
    includes Levene and Brown-Forsythe tests for equal variances as special
    cases.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2. The data can be provided
        as a tuple or list of arrays or in long format with outcome
        observations in ``data`` and group membership in ``groups``.
    method : {"unequal", "equal" or "bf"}
        How to treat heteroscedasticity across samples. This is used as
        `use_var` option in `anova_oneway` and refers to the variance of the
        transformed data, i.e. assumption is on 4th moment if squares are used
        as transform.
        Three approaches are available:

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : "abs", "square" or callable
        Transformation for the centered observations. If a callable, then this
        function is called on the centered data.
        Default is absolute value.
    trim_frac_mean=0.1 : float in [0, 0.5)
        Trim fraction for the trimmed mean when `center` is "trimmed"
    trim_frac_anova : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and Winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        see ``trim_frac`` option in `anova_oneway`

    Returns
    -------
    res : results instance
        The returned HolderTuple instance has the following main attributes
        and some additional information in other attributes.

        statistic : float
            Test statistic for k-sample mean comparison which is approximately
            F-distributed.
        pvalue : float
            If ``method="bf"``, then the p-value is based on corrected
            degrees of freedom following Mehrotra 1997.
        pvalue2 : float
            This is the p-value based on degrees of freedom as in
            Brown-Forsythe 1974 and is only available if ``method="bf"``.
        df : (df_denom, df_num)
            Tuple containing degrees of freedom for the F-distribution depend
            on ``method``. If ``method="bf"``, then `df_denom` is for Mehrotra
            p-values `df_denom2` is available for Brown-Forsythe 1974 p-values.
            `df_num` is the same numerator degrees of freedom for both
            p-values.

    See Also
    --------
    anova_oneway
    scale_transform
    """
    data = map(np.asarray, data)
    xxd = [scale_transform(x, center=center, transform=transform, trim_frac=trim_frac_mean) for x in data]
    res = anova_oneway(xxd, groups=None, use_var=method, welch_correction=True, trim_frac=trim_frac_anova)
    res.data_transformed = xxd
    return res