import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_simple_object_listing():
    assert isinstance(central.select.projects().get(), list)