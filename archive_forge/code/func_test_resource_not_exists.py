import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_resource_not_exists():
    assert not resource_2.exists()
    assert isinstance(resource_2, object)
    assert isinstance(resource_2, pyxnat.core.resources.Resource)
    assert str(resource_2) == '<Resource Object> IOP'