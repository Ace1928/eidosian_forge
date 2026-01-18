import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('is_method', (True, False))
def test_wrap_r_function(is_method):
    r_code = 'function(x, y=FALSE, z="abc") TRUE'
    parameter_names = ('self', 'x', 'y', 'z') if is_method else ('x', 'y', 'z')
    r_func = robjects.r(r_code)
    foo = robjects.functions.wrap_r_function(r_func, 'foo', is_method=is_method)
    assert inspect.getclosurevars(foo).nonlocals['r_func'].rid == r_func.rid
    assert tuple(foo.__signature__.parameters.keys()) == parameter_names
    if not is_method:
        res = foo(1)
        assert res[0] is True