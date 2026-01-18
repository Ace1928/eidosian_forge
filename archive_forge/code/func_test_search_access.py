import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_search_access():
    constraints = [('xnat:subjectData/PROJECT', '=', 'ixi'), 'AND']
    for subject in central.select('//subjects').where(constraints):
        assert '/projects/ixi' in subject._uri