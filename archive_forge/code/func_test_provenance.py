import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_provenance(tmpdir):
    metawf = pe.Workflow(name='meta')
    metawf.base_dir = tmpdir.strpath
    metawf.add_nodes([create_wf('wf%d' % i) for i in range(1)])
    eg = metawf.run(plugin='Linear')
    prov_base = tmpdir.join('workflow_provenance_test').strpath
    psg = write_workflow_prov(eg, prov_base, format='all')
    assert len(psg.bundles) == 2
    assert len(psg.get_records()) == 7