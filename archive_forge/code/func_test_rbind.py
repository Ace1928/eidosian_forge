import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_rbind():
    dataf = robjects.r('data.frame(a=1:2, b=I(c("a", "b")))')
    dataf = dataf.rbind(dataf)
    assert dataf.ncol == 2
    assert dataf.nrow == 4