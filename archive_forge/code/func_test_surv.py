import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest
def test_surv():
    np.random.seed(2341)
    n = 1000
    exp = np.random.normal(size=n)
    mn = np.exp(exp)
    mtime0 = -mn * np.log(np.random.uniform(size=n))
    ctime = -2 * mn * np.log(np.random.uniform(size=n))
    mstatus = (ctime >= mtime0).astype(int)
    mtime = np.where(mtime0 <= ctime, mtime0, ctime)
    for mt in ('full', 'partial', 'no'):
        if mt == 'full':
            lp = 0.5 * mtime0
        elif mt == 'partial':
            lp = exp + mtime0
        else:
            lp = exp
        mn = np.exp(-lp)
        ytime0 = -mn * np.log(np.random.uniform(size=n))
        ctime = -2 * mn * np.log(np.random.uniform(size=n))
        ystatus = (ctime >= ytime0).astype(int)
        ytime = np.where(ytime0 <= ctime, ytime0, ctime)
        df = pd.DataFrame({'ytime': ytime, 'ystatus': ystatus, 'mtime': mtime, 'mstatus': mstatus, 'exp': exp})
        fml = 'ytime ~ exp + mtime'
        outcome_model = sm.PHReg.from_formula(fml, status='ystatus', data=df)
        fml = 'mtime ~ exp'
        mediator_model = sm.PHReg.from_formula(fml, status='mstatus', data=df)
        med = Mediation(outcome_model, mediator_model, 'exp', 'mtime', outcome_predict_kwargs={'pred_only': True}, outcome_fit_kwargs={'method': 'lbfgs'}, mediator_fit_kwargs={'method': 'lbfgs'})
        med_result = med.fit(n_rep=2)
        dr = med_result.summary()
        pm = dr.loc['Prop. mediated (average)', 'Estimate']
        if mt == 'no':
            assert_allclose(pm, 0, atol=0.1, rtol=0.1)
        elif mt == 'full':
            assert_allclose(pm, 1, atol=0.1, rtol=0.1)
        else:
            assert_allclose(pm, 0.5, atol=0.1, rtol=0.1)