import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
def test_ri2pandas_issue207(self):
    d = robjects.DataFrame({'x': 1})
    with localconverter(default_converter + rpyp.converter) as cv:
        try:
            ok = True
            robjects.globalenv['d'] = d
        except ValueError:
            ok = False
        finally:
            if 'd' in robjects.globalenv:
                del robjects.globalenv['d']
    assert ok