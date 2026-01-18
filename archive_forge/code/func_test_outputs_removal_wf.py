from glob import glob
import os
from shutil import rmtree
from itertools import product
import pytest
import networkx as nx
from .... import config
from ....interfaces import utility as niu
from ... import engine as pe
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
@pytest.mark.parametrize('plugin, remove_unnecessary_outputs, keep_inputs', list(product(['Linear', 'MultiProc'], [False, True], [True, False])))
def test_outputs_removal_wf(tmpdir, plugin, remove_unnecessary_outputs, keep_inputs):
    config.set_default_config()
    config.set('execution', 'remove_unnecessary_outputs', remove_unnecessary_outputs)
    config.set('execution', 'keep_inputs', keep_inputs)
    n1 = pe.Node(niu.Function(output_names=['out_file1', 'out_file2', 'dir'], function=_test_function), name='n1', base_dir=tmpdir.strpath)
    n1.inputs.arg1 = 1
    n2 = pe.Node(niu.Function(output_names=['out_file1', 'out_file2', 'n'], function=_test_function2), name='n2', base_dir=tmpdir.strpath)
    n2.inputs.arg = 2
    n3 = pe.Node(niu.Function(output_names=['n'], function=_test_function3), name='n3', base_dir=tmpdir.strpath)
    wf = pe.Workflow(name='node_rem_test' + plugin, base_dir=tmpdir.strpath)
    wf.connect(n1, 'out_file1', n2, 'in_file')
    wf.run(plugin=plugin)
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n1.name, 'file1.txt'))
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n2.name, 'file1.txt'))
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n2.name, 'file2.txt'))
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n1.name, 'file2.txt')) is not remove_unnecessary_outputs
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n1.name, 'subdir', 'file4.txt')) is not remove_unnecessary_outputs
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n1.name, 'file3.txt')) is not remove_unnecessary_outputs
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n2.name, 'file3.txt')) is not remove_unnecessary_outputs
    n4 = pe.Node(UtilsTestInterface(), name='n4', base_dir=tmpdir.strpath)
    wf.connect(n2, 'out_file1', n4, 'in_file')

    def pick_first(l):
        return l[0]
    wf.connect(n4, ('output1', pick_first), n3, 'arg')
    rmtree(os.path.join(wf.base_dir, wf.name))
    wf.run(plugin=plugin)
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n2.name, 'file1.txt'))
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n2.name, 'file1.txt'))
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n2.name, 'file2.txt')) is not remove_unnecessary_outputs
    assert os.path.exists(os.path.join(wf.base_dir, wf.name, n4.name, 'file1.txt')) is keep_inputs