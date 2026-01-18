from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread, imsave, plugin_order
from skimage._shared import testing
from skimage._shared.testing import fetch, assert_stacklevel
import pytest
def test_prefered_plugin():
    order = plugin_order()
    assert order['imread'][0] == 'imageio'
    assert order['imsave'][0] == 'imageio'
    assert order['imread_collection'][0] == 'imageio'