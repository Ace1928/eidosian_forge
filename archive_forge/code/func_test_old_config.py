from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_old_config(tmpdir):
    tmpdir.chdir()
    wd = os.getcwd()
    from nipype.interfaces.utility import Function

    def func1():
        return 1

    def func2(a):
        return a + 1
    n1 = pe.Node(Function(input_names=[], output_names=['a'], function=func1), name='n1')
    n2 = pe.Node(Function(input_names=['a'], output_names=['b'], function=func2), name='n2')
    w1 = pe.Workflow(name='test')
    modify = lambda x: x + 1
    n1.inputs.a = 1
    w1.connect(n1, ('a', modify), n2, 'a')
    w1.base_dir = wd
    w1.config['execution']['crashdump_dir'] = wd
    w1.run(plugin='Linear')