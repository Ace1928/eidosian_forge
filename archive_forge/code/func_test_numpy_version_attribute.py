import os
import pytest
import numpy as np
from . import util
@pytest.mark.slow
def test_numpy_version_attribute(self):
    assert hasattr(self.module, '__f2py_numpy_version__')
    assert isinstance(self.module.__f2py_numpy_version__, str)
    assert np.__version__ == self.module.__f2py_numpy_version__