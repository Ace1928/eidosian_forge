from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_write_graph_runs(tmpdir):
    tmpdir.chdir()
    for graph in ('orig', 'flat', 'exec', 'hierarchical', 'colored'):
        for simple in (True, False):
            pipe = pe.Workflow(name='pipe')
            mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
            mod2 = pe.Node(interface=EngineTestInterface(), name='mod2')
            pipe.connect([(mod1, mod2, [('output1', 'input1')])])
            try:
                pipe.write_graph(graph2use=graph, simple_form=simple, format='dot')
            except Exception:
                assert False, 'Failed to plot {} {} graph'.format('simple' if simple else 'detailed', graph)
            assert os.path.exists('graph.dot') or os.path.exists('graph_detailed.dot')
            try:
                os.remove('graph.dot')
            except OSError:
                pass
            try:
                os.remove('graph_detailed.dot')
            except OSError:
                pass