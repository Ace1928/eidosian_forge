import pytest
from statsmodels.tools.docstring import Docstring, remove_parameters, Parameter
def test_remove_parameter():
    ds = Docstring(good)
    ds.remove_parameters('x')
    assert 'x : int' not in str(ds)
    ds = Docstring(good)
    ds.remove_parameters(['x', 'y'])
    assert 'x : int' not in str(ds)
    assert 'y : float' not in str(ds)
    with pytest.raises(ValueError):
        Docstring(good).remove_parameters(['w'])
    ds = remove_parameters(good, 'x')
    assert 'x : int' not in ds
    assert isinstance(ds, str)