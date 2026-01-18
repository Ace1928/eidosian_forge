import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test_te_errors():
    import pytest
    x = np.arange(27)
    pytest.raises(ValueError, te, x.reshape((3, 3, 3)))
    pytest.raises(ValueError, te, x.reshape((3, 3, 3)), constraints='center')
    pytest.raises(ValueError, te, x, constraints=np.arange(8).reshape((2, 2, 2)))