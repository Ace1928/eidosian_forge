import numpy as np
import pytest
from opt_einsum import contract, contract_path
def test_value_errors():
    with pytest.raises(ValueError):
        contract('')
    with pytest.raises(TypeError):
        contract(0, 0)
    with pytest.raises(ValueError):
        contract('i%...', [0, 0])
    with pytest.raises(ValueError):
        contract('...j$', [0, 0])
    with pytest.raises(ValueError):
        contract('i->&', [0, 0])
    with pytest.raises(ValueError):
        contract('')
    with pytest.raises(ValueError):
        contract('', 0, 0)
    with pytest.raises(ValueError):
        contract(',', 0, [0], [0])
    with pytest.raises(ValueError):
        contract(',', [0])
    with pytest.raises(ValueError):
        contract('i', 0)
    with pytest.raises(ValueError):
        contract('ij', [0, 0])
    with pytest.raises(ValueError):
        contract('...i', 0)
    with pytest.raises(ValueError):
        contract('i...j', [0, 0])
    with pytest.raises(ValueError):
        contract('i...', 0)
    with pytest.raises(ValueError):
        contract('ij...', [0, 0])
    with pytest.raises(ValueError):
        contract('i..', [0, 0])
    with pytest.raises(ValueError):
        contract('.i...', [0, 0])
    with pytest.raises(ValueError):
        contract('j->..j', [0, 0])
    with pytest.raises(ValueError):
        contract('j->.j...', [0, 0])
    with pytest.raises(ValueError):
        contract('i%...', [0, 0])
    with pytest.raises(ValueError):
        contract('...j$', [0, 0])
    with pytest.raises(ValueError):
        contract('i->&', [0, 0])
    with pytest.raises(ValueError):
        contract('i->ij', [0, 0])
    with pytest.raises(ValueError):
        contract('ij->jij', [[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        contract('ii', np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract('ii->i', np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract('i', np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract('i->i', [[0, 1], [0, 1]], out=np.arange(4).reshape(2, 2))