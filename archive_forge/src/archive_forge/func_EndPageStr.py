import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def EndPageStr(self):
    self.inPageFlag = 0
    return ''