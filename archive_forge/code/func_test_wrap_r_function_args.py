import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('r_code,args,kwargs,expected', (('function(x, y=1, z=2) {sum(x, y, z)}', (3,), {}, 6), ('function(x, y=1, z=2) {sum(x, y, z)}', (3, 4), {}, 9), ('function(...) {sum(...)}', (3, 2, 4), {}, 9), ('function(x, ...) {sum(x, ...)}', (3, 2, 4), {}, 9), ('function(x, ..., z=1) {sum(x, ..., z)}', (3, 2, 4), {}, 10), ('function(x, ..., z=1) {sum(x, ..., z)}', (3, 2, 4), {'z': 2}, 11)))
def test_wrap_r_function_args(r_code, args, kwargs, expected):
    r_func = robjects.r(r_code)
    stf = robjects.functions.SignatureTranslatedFunction(r_func)
    w_func = robjects.functions.wrap_r_function(stf, 'foo')
    res = w_func(*args, **kwargs)
    assert tuple(res) == (expected,)