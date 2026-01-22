import numpy as np
import skimage.io as io
from skimage._shared import testing
@testing.pytest.fixture(autouse=True)
def _reset_plugin():
    yield
    io.reset_plugins()