import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_set_join_node(tmpdir):
    """Test collecting join inputs to a set."""
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [1, 2, 1, 3, 2])]
    pre_join1 = pe.Node(IncrementInterface(), name='pre_join1')
    wf.connect(inputspec, 'n', pre_join1, 'input1')
    join = pe.JoinNode(SetInterface(), joinsource='inputspec', joinfield='input1', name='join')
    wf.connect(pre_join1, 'output1', join, 'input1')
    wf.run()
    assert _set_len == 3, 'The join Set output value is incorrect: %s.' % _set_len