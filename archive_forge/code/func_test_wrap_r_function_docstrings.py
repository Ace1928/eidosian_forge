import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('wrap_docstring', (None, robjects.functions.wrap_docstring_default))
def test_wrap_r_function_docstrings(wrap_docstring):
    r_code = 'function(x, y=FALSE, z="abc") TRUE'
    r_func = robjects.r(r_code)
    foo = robjects.functions.wrap_r_function(r_func, 'foo', wrap_docstring=wrap_docstring)