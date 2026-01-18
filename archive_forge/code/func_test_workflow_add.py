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
def test_workflow_add():
    n1 = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='n1')
    n2 = pe.Node(niu.IdentityInterface(fields=['c', 'd']), name='n2')
    n3 = pe.Node(niu.IdentityInterface(fields=['c', 'd']), name='n1')
    w1 = pe.Workflow(name='test')
    w1.connect(n1, 'a', n2, 'c')
    for node in [n1, n2, n3]:
        with pytest.raises(IOError):
            w1.add_nodes([node])
    with pytest.raises(IOError):
        w1.connect([(w1, n2, [('n1.a', 'd')])])