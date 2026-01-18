import pytest
from statsmodels.tools.docstring import Docstring, remove_parameters, Parameter
def test_yield_return():
    with pytest.raises(ValueError):
        Docstring(bad_yields)