import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_sexp_rsame_false():
    sexp_a = rinterface.baseenv.find('letters')
    sexp_b = rinterface.baseenv.find('pi')
    assert not sexp_a.rsame(sexp_b)