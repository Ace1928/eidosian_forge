import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_to_csvfile():
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fh.close()
    d = {'letter': robjects.StrVector('abc'), 'value': robjects.IntVector((1, 2, 3))}
    dataf = robjects.DataFrame(d)
    dataf.to_csvfile(fh.name)
    dataf = robjects.DataFrame.from_csvfile(fh.name)
    assert dataf.nrow == 3
    assert dataf.ncol == 2