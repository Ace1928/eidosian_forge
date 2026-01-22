from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
def _resid_info(self):
    buf = StringIO()
    names = self.model.names
    buf.write('Correlation matrix of residuals' + '\n')
    buf.write(pprint_matrix(self.model.resid_corr, names, names) + '\n')
    return buf.getvalue()