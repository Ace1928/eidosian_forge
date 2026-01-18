import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_rclass_get():
    sexp = rinterface.baseenv.find('character')(1)
    assert len(sexp.rclass) == 1
    assert sexp.rclass[0] == 'character'
    sexp = rinterface.baseenv.find('matrix')(0)
    if rinterface.evalr('R.version$major')[0] >= '4':
        assert tuple(sexp.rclass) == ('matrix', 'array')
    else:
        assert tuple(sexp.rclass) == ('matrix',)
    sexp = rinterface.baseenv.find('array')(0)
    assert len(sexp.rclass) == 1
    assert sexp.rclass[0] == 'array'
    sexp = rinterface.baseenv.find('new.env')()
    assert len(sexp.rclass) == 1
    assert sexp.rclass[0] == 'environment'