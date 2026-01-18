from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
@pytest.mark.parametrize('simple', [True, False])
@pytest.mark.parametrize('graph_type', ['orig', 'flat', 'exec', 'hierarchical', 'colored'])
def test_write_graph_dotfile(tmpdir, graph_type, simple):
    """checking dot files for a workflow without iterables"""
    tmpdir.chdir()
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
    mod2 = pe.Node(interface=EngineTestInterface(), name='mod2')
    pipe.connect([(mod1, mod2, [('output1', 'input1')])])
    pipe.write_graph(graph2use=graph_type, simple_form=simple, format='dot')
    with open('graph.dot') as f:
        graph_str = f.read()
    if simple:
        for line in dotfiles[graph_type]:
            assert line in graph_str
    else:
        for line in dotfiles[graph_type]:
            if graph_type in ['hierarchical', 'colored']:
                assert line.replace('mod1 (engine)', 'mod1.EngineTestInterface.engine').replace('mod2 (engine)', 'mod2.EngineTestInterface.engine') in graph_str
            else:
                assert line.replace('mod1 (engine)', 'pipe.mod1.EngineTestInterface.engine').replace('mod2 (engine)', 'pipe.mod2.EngineTestInterface.engine') in graph_str
    if graph_type not in ['hierarchical', 'colored']:
        with open('graph_detailed.dot') as f:
            graph_str = f.read()
        for line in dotfile_detailed_orig:
            assert line in graph_str