from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_parameterize_dirs_false(tmpdir):
    from ....interfaces.utility import IdentityInterface
    from ....testing import example_data
    input_file = example_data('fsl_motion_outliers_fd.txt')
    n1 = pe.Node(EngineTestInterface(), name='Node1')
    n1.iterables = ('input_file', (input_file, input_file))
    n1.interface.inputs.input1 = 1
    n2 = pe.Node(IdentityInterface(fields='in1'), name='Node2')
    wf = pe.Workflow(name='Test')
    wf.base_dir = tmpdir.strpath
    wf.config['execution']['parameterize_dirs'] = False
    wf.connect([(n1, n2, [('output1', 'in1')])])
    wf.run()