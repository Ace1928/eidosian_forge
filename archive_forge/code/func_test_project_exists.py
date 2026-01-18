import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_project_exists():
    assert proj_1.exists()
    assert isinstance(proj_1, object)
    assert isinstance(proj_1, pyxnat.core.resources.Project)
    assert str(proj_1) != '<Project Object> NFB'