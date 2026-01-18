import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_rclass_get_sym():
    fit = rinterface.evalr('\n    stats::lm(y ~ x, data=base::data.frame(y=1:10, x=2:11))\n    ')
    assert tuple(fit[9].rclass) == ('call',)