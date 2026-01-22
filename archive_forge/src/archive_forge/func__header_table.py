from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
def _header_table(self):
    import time
    model = self.model
    t = time.localtime()
    part1title = 'Summary of Regression Results'
    part1data = [[model._model_type], ['OLS'], [time.strftime('%a, %d, %b, %Y', t)], [time.strftime('%H:%M:%S', t)]]
    part1header = None
    part1stubs = ('Model:', 'Method:', 'Date:', 'Time:')
    part1 = SimpleTable(part1data, part1header, part1stubs, title=part1title, txt_fmt=self.part1_fmt)
    return str(part1)