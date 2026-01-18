import pytest
import numpy as np
import ase.io.ulm as ulm
@pytest.fixture
def ulmfile(tmp_path):
    path = tmp_path / 'a.ulm'
    with ulm.open(path, 'w') as w:
        w.write(a=A(), y=9)
        w.write(s='abc')
        w.sync()
        w.write(s='abc2')
        w.sync()
        w.write(s='abc3', z=np.ones(7, int))
    return path