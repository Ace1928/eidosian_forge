import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_subject_exists():
    assert subj_1.exists()
    assert isinstance(subj_1, object)
    assert isinstance(subj_1, pyxnat.core.resources.Subject)
    assert str(subj_1) != '<Subject Object> BOA'