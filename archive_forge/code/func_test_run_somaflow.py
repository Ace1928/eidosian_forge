import os
from time import sleep
import nipype.interfaces.base as nib
import pytest
import nipype.pipeline.engine as pe
from nipype.pipeline.plugins.somaflow import soma_not_loaded
@pytest.mark.skipif(soma_not_loaded, reason='soma not loaded')
def test_run_somaflow(tmpdir):
    tmpdir.chdir()
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(interface=SomaTestInterface(), name='mod1')
    mod2 = pe.MapNode(interface=SomaTestInterface(), iterfield=['input1'], name='mod2')
    pipe.connect([(mod1, mod2, [('output1', 'input1')])])
    pipe.base_dir = os.getcwd()
    mod1.inputs.input1 = 1
    execgraph = pipe.run(plugin='SomaFlow')
    names = ['.'.join((node._hierarchy, node.name)) for node in execgraph.nodes()]
    node = list(execgraph.nodes())[names.index('pipe.mod1')]
    result = node.get_output('output1')
    assert result == [1, 1]