import os
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.regression.linear_model import OLS, GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
import statsmodels.stats.sandwich_covariance as sw
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
class TestGLSARGretl:

    def test_all(self):
        d = macrodata.load_pandas().data
        gs_l_realinv = 400 * np.diff(np.log(d['realinv'].values))
        gs_l_realgdp = 400 * np.diff(np.log(d['realgdp'].values))
        endogd = np.diff(d['realinv'])
        exogd = add_constant(np.c_[np.diff(d['realgdp'].values), d['realint'][:-1].values])
        endogg = gs_l_realinv
        exogg = add_constant(np.c_[gs_l_realgdp, d['realint'][:-1].values])
        res_ols = OLS(endogg, exogg).fit()
        mod_g1 = GLSAR(endogg, exogg, rho=-0.108136)
        res_g1 = mod_g1.fit()
        mod_g2 = GLSAR(endogg, exogg, rho=-0.108136)
        res_g2 = mod_g2.iterative_fit(maxiter=5)
        rho = -0.108136
        partable = np.array([[-9.5099, 0.990456, -9.602, 3.65e-18, -11.4631, -7.5567], [4.3704, 0.208146, 21.0, 2.93e-52, 3.95993, 4.78086], [-0.579253, 0.268009, -2.161, 0.0319, -1.10777, -0.0507346]])
        result_gretl_g1 = dict(endog_mean=('Mean dependent var', 3.113973), endog_std=('S.D. dependent var', 18.67447), ssr=('Sum squared resid', 22530.9), mse_resid_sqrt=('S.E. of regression', 10.66735), rsquared=('R-squared', 0.676973), rsquared_adj=('Adjusted R-squared', 0.67371), fvalue=('F(2, 198)', 221.0475), f_pvalue=('P-value(F)', 3.56e-51), resid_acf1=('rho', -0.003481), dw=('Durbin-Watson', 1.993858))
        reset_2_3 = [5.219019, 0.00619, 2, 197, 'f']
        reset_2 = [7.268492, 0.00762, 1, 198, 'f']
        reset_3 = [5.248951, 0.023, 1, 198, 'f']
        arch_4 = [7.30776, 0.120491, 4, 'chi2']
        vif = [1.002, 1.002]
        cond_1norm = 6862.0664
        determinant = 1029604900.0
        reciprocal_condition_number = 0.013819244
        normality = [20.2792, 3.94837e-05, 2]
        res = res_g1
        assert_almost_equal(res.params, partable[:, 0], 4)
        assert_almost_equal(res.bse, partable[:, 1], 6)
        assert_almost_equal(res.tvalues, partable[:, 2], 2)
        assert_almost_equal(res.ssr, result_gretl_g1['ssr'][1], decimal=2)
        assert_almost_equal(np.sqrt(res.mse_resid), result_gretl_g1['mse_resid_sqrt'][1], decimal=5)
        assert_almost_equal(res.fvalue, result_gretl_g1['fvalue'][1], decimal=4)
        assert_allclose(res.f_pvalue, result_gretl_g1['f_pvalue'][1], rtol=0.01)
        sm_arch = smsdia.het_arch(res.wresid, nlags=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=4)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=6)
        res = res_g2
        assert_almost_equal(res.model.rho, rho, decimal=3)
        assert_almost_equal(res.params, partable[:, 0], 4)
        assert_almost_equal(res.bse, partable[:, 1], 3)
        assert_almost_equal(res.tvalues, partable[:, 2], 2)
        assert_almost_equal(res.ssr, result_gretl_g1['ssr'][1], decimal=2)
        assert_almost_equal(np.sqrt(res.mse_resid), result_gretl_g1['mse_resid_sqrt'][1], decimal=5)
        assert_almost_equal(res.fvalue, result_gretl_g1['fvalue'][1], decimal=0)
        assert_almost_equal(res.f_pvalue, result_gretl_g1['f_pvalue'][1], decimal=6)
        c = oi.reset_ramsey(res, degree=2)
        compare_ftest(c, reset_2, decimal=(2, 4))
        c = oi.reset_ramsey(res, degree=3)
        compare_ftest(c, reset_2_3, decimal=(2, 4))
        sm_arch = smsdia.het_arch(res.wresid, nlags=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=1)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=2)
        '\n        Performing iterative calculation of rho...\n\n                         ITER       RHO        ESS\n                           1     -0.10734   22530.9\n                           2     -0.10814   22530.9\n\n        Model 4: Cochrane-Orcutt, using observations 1959:3-2009:3 (T = 201)\n        Dependent variable: ds_l_realinv\n        rho = -0.108136\n\n                         coefficient   std. error   t-ratio    p-value\n          -------------------------------------------------------------\n          const           -9.50990      0.990456    -9.602    3.65e-018 ***\n          ds_l_realgdp     4.37040      0.208146    21.00     2.93e-052 ***\n          realint_1       -0.579253     0.268009    -2.161    0.0319    **\n\n        Statistics based on the rho-differenced data:\n\n        Mean dependent var   3.113973   S.D. dependent var   18.67447\n        Sum squared resid    22530.90   S.E. of regression   10.66735\n        R-squared            0.676973   Adjusted R-squared   0.673710\n        F(2, 198)            221.0475   P-value(F)           3.56e-51\n        rho                 -0.003481   Durbin-Watson        1.993858\n        '
        '\n        RESET test for specification (squares and cubes)\n        Test statistic: F = 5.219019,\n        with p-value = P(F(2,197) > 5.21902) = 0.00619\n\n        RESET test for specification (squares only)\n        Test statistic: F = 7.268492,\n        with p-value = P(F(1,198) > 7.26849) = 0.00762\n\n        RESET test for specification (cubes only)\n        Test statistic: F = 5.248951,\n        with p-value = P(F(1,198) > 5.24895) = 0.023:\n        '
        '\n        Test for ARCH of order 4\n\n                     coefficient   std. error   t-ratio   p-value\n          --------------------------------------------------------\n          alpha(0)   97.0386       20.3234       4.775    3.56e-06 ***\n          alpha(1)    0.176114      0.0714698    2.464    0.0146   **\n          alpha(2)   -0.0488339     0.0724981   -0.6736   0.5014\n          alpha(3)   -0.0705413     0.0737058   -0.9571   0.3397\n          alpha(4)    0.0384531     0.0725763    0.5298   0.5968\n\n          Null hypothesis: no ARCH effect is present\n          Test statistic: LM = 7.30776\n          with p-value = P(Chi-square(4) > 7.30776) = 0.120491:\n        '
        "\n        Variance Inflation Factors\n\n        Minimum possible value = 1.0\n        Values > 10.0 may indicate a collinearity problem\n\n           ds_l_realgdp    1.002\n              realint_1    1.002\n\n        VIF(j) = 1/(1 - R(j)^2), where R(j) is the multiple correlation coefficient\n        between variable j and the other independent variables\n\n        Properties of matrix X'X:\n\n         1-norm = 6862.0664\n         Determinant = 1.0296049e+009\n         Reciprocal condition number = 0.013819244\n        "
        '\n        Test for ARCH of order 4 -\n          Null hypothesis: no ARCH effect is present\n          Test statistic: LM = 7.30776\n          with p-value = P(Chi-square(4) > 7.30776) = 0.120491\n\n        Test of common factor restriction -\n          Null hypothesis: restriction is acceptable\n          Test statistic: F(2, 195) = 0.426391\n          with p-value = P(F(2, 195) > 0.426391) = 0.653468\n\n        Test for normality of residual -\n          Null hypothesis: error is normally distributed\n          Test statistic: Chi-square(2) = 20.2792\n          with p-value = 3.94837e-005:\n        '
        '\n        Augmented regression for common factor test\n        OLS, using observations 1959:3-2009:3 (T = 201)\n        Dependent variable: ds_l_realinv\n\n                           coefficient   std. error   t-ratio    p-value\n          ---------------------------------------------------------------\n          const            -10.9481      1.35807      -8.062    7.44e-014 ***\n          ds_l_realgdp       4.28893     0.229459     18.69     2.40e-045 ***\n          realint_1         -0.662644    0.334872     -1.979    0.0492    **\n          ds_l_realinv_1    -0.108892    0.0715042    -1.523    0.1294\n          ds_l_realgdp_1     0.660443    0.390372      1.692    0.0923    *\n          realint_2          0.0769695   0.341527      0.2254   0.8219\n\n          Sum of squared residuals = 22432.8\n\n        Test of common factor restriction\n\n          Test statistic: F(2, 195) = 0.426391, with p-value = 0.653468\n        '
        partable = np.array([[-9.48167, 1.17709, -8.055, 7.17e-14, -11.8029, -7.16049], [4.37422, 0.328787, 13.3, 2.62e-29, 3.72587, 5.02258], [-0.613997, 0.293619, -2.091, 0.0378, -1.193, -0.0349939]])
        result_gretl_g1 = dict(endog_mean=('Mean dependent var', 3.257395), endog_std=('S.D. dependent var', 18.73915), ssr=('Sum squared resid', 22799.68), mse_resid_sqrt=('S.E. of regression', 10.7038), rsquared=('R-squared', 0.676978), rsquared_adj=('Adjusted R-squared', 0.673731), fvalue=('F(2, 199)', 90.79971), f_pvalue=('P-value(F)', 9.53e-29), llf=('Log-likelihood', -763.9752), aic=('Akaike criterion', 1533.95), bic=('Schwarz criterion', 1543.875), hqic=('Hannan-Quinn', 1537.966), resid_acf1=('rho', -0.107341), dw=('Durbin-Watson', 2.213805))
        linear_logs = [1.68351, 0.430953, 2, 'chi2']
        linear_squares = [7.52477, 0.0232283, 2, 'chi2']
        lm_acorr4 = [1.17928, 0.321197, 4, 195, 'F']
        lm2_acorr4 = [4.771043, 0.312, 4, 'chi2']
        acorr_ljungbox4 = [5.23587, 0.264, 4, 'chi2']
        cusum_Harvey_Collier = [0.494432, 0.621549, 198, 't']
        break_qlr = [3.01985, 0.1, 3, 196, 'maxF']
        break_chow = [13.1897, 0.00424384, 3, 'chi2']
        arch_4 = [3.43473, 0.487871, 4, 'chi2']
        normality = [23.962, 1e-05, 2, 'chi2']
        het_white = [33.503723, 3e-06, 5, 'chi2']
        het_breusch_pagan = [1.302014, 0.52152, 2, 'chi2']
        het_breusch_pagan_konker = [0.709924, 0.7012, 2, 'chi2']
        reset_2_3 = [5.219019, 0.00619, 2, 197, 'f']
        reset_2 = [7.268492, 0.00762, 1, 198, 'f']
        reset_3 = [5.248951, 0.023, 1, 198, 'f']
        cond_1norm = 5984.0525
        determinant = 710874670.0
        reciprocal_condition_number = 0.013826504
        vif = [1.001, 1.001]
        names = 'date   residual        leverage       influence        DFFITS'.split()
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        fpath = os.path.join(cur_dir, 'results/leverage_influence_ols_nostars.txt')
        lev = np.genfromtxt(fpath, skip_header=3, skip_footer=1, converters={0: lambda s: s})
        if np.isnan(lev[-1]['f1']):
            lev = np.genfromtxt(fpath, skip_header=3, skip_footer=2, converters={0: lambda s: s})
        lev.dtype.names = names
        res = res_ols
        cov_hac = sw.cov_hac_simple(res, nlags=4, use_correction=False)
        bse_hac = sw.se_cov(cov_hac)
        assert_almost_equal(res.params, partable[:, 0], 5)
        assert_almost_equal(bse_hac, partable[:, 1], 5)
        assert_almost_equal(res.ssr, result_gretl_g1['ssr'][1], decimal=2)
        assert_almost_equal(res.llf, result_gretl_g1['llf'][1], decimal=4)
        assert_almost_equal(res.rsquared, result_gretl_g1['rsquared'][1], decimal=6)
        assert_almost_equal(res.rsquared_adj, result_gretl_g1['rsquared_adj'][1], decimal=6)
        assert_almost_equal(np.sqrt(res.mse_resid), result_gretl_g1['mse_resid_sqrt'][1], decimal=5)
        c = oi.reset_ramsey(res, degree=2)
        compare_ftest(c, reset_2, decimal=(6, 5))
        c = oi.reset_ramsey(res, degree=3)
        compare_ftest(c, reset_2_3, decimal=(6, 5))
        linear_sq = smsdia.linear_lm(res.resid, res.model.exog)
        assert_almost_equal(linear_sq[0], linear_squares[0], decimal=6)
        assert_almost_equal(linear_sq[1], linear_squares[1], decimal=7)
        hbpk = smsdia.het_breuschpagan(res.resid, res.model.exog)
        assert_almost_equal(hbpk[0], het_breusch_pagan_konker[0], decimal=6)
        assert_almost_equal(hbpk[1], het_breusch_pagan_konker[1], decimal=6)
        hw = smsdia.het_white(res.resid, res.model.exog)
        assert_almost_equal(hw[:2], het_white[:2], 6)
        sm_arch = smsdia.het_arch(res.resid, nlags=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=5)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=6)
        vif2 = [oi.variance_inflation_factor(res.model.exog, k) for k in [1, 2]]
        infl = oi.OLSInfluence(res_ols)
        assert_almost_equal(lev['residual'], res.resid, decimal=3)
        assert_almost_equal(lev['DFFITS'], infl.dffits[0], decimal=3)
        assert_almost_equal(lev['leverage'], infl.hat_matrix_diag, decimal=3)
        assert_almost_equal(lev['influence'], infl.influence, decimal=4)