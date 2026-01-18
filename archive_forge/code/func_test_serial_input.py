from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_serial_input(tmpdir):
    tmpdir.chdir()
    wd = os.getcwd()
    from nipype import MapNode, Function, Workflow

    def func1(in1):
        return in1
    n1 = MapNode(Function(input_names=['in1'], output_names=['out'], function=func1), iterfield=['in1'], name='n1')
    n1.inputs.in1 = [1, 2, 3]
    w1 = Workflow(name='test')
    w1.base_dir = wd
    w1.add_nodes([n1])
    w1.config['execution'] = {'stop_on_first_crash': 'true', 'local_hash_check': 'true', 'crashdump_dir': wd, 'poll_sleep_duration': 2}
    assert n1.num_subnodes() == len(n1.inputs.in1)
    w1.run(plugin='MultiProc')
    n1._serial = True
    assert n1.num_subnodes() == 1
    w1.run(plugin='MultiProc')