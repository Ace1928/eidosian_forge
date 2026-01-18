import numpy as np
from numpy.testing import assert_, assert_almost_equal
from statsmodels.sandbox.distributions.extras import (skewnorm,
def test_skewnorm():
    pdf_r = np.array([2.973416551551523e-90, 3.687562713971017e-24, 0.3989422804014327, 0.4839414490382867, 0.1079819330263761])
    pdf_sn = skewnorm.pdf([-2, -1, 0, 1, 2], 10)
    assert_(np.allclose(pdf_sn, pdf_r, rtol=1e-13, atol=0))
    pdf_sn2 = skewnorm2.pdf([-2, -1, 0, 1, 2], 10)
    assert_(np.allclose(pdf_sn2, pdf_r, rtol=1e-13, atol=0))
    cdf_r = np.array([0.0, 0.0, 0.03172551743055357, 0.6826894921370859, 0.9544997361036416])
    cdf_sn = skewnorm.cdf([-2, -1, 0, 1, 2], 10)
    maxabs = np.max(np.abs(cdf_sn - cdf_r))
    maxrel = np.max(np.abs(cdf_sn - cdf_r) / (cdf_r + 1e-50))
    msg = 'maxabs={:15.13g}, maxrel={:15.13g}\n{!r}\n{!r}'.format(maxabs, maxrel, cdf_sn, cdf_r)
    assert_almost_equal(cdf_sn, cdf_r, decimal=10)
    cdf_sn2 = skewnorm2.cdf([-2, -1, 0, 1, 2], 10)
    maxabs = np.max(np.abs(cdf_sn2 - cdf_r))
    maxrel = np.max(np.abs(cdf_sn2 - cdf_r) / (cdf_r + 1e-50))
    msg = 'maxabs={:15.13g}, maxrel={:15.13g}'.format(maxabs, maxrel)
    assert_almost_equal(cdf_sn2, cdf_r, decimal=10, err_msg=msg)