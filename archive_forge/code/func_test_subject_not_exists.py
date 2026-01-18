import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_subject_not_exists():
    assert not subj_2.exists()
    assert isinstance(subj_2, object)
    assert isinstance(subj_2, pyxnat.core.resources.Subject)
    assert str(subj_2) == '<Subject Object> BOA'