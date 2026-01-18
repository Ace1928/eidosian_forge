import os
import nipype.interfaces.base as nib
import nipype.pipeline.engine as pe
def test_run_in_series(tmpdir):
    tmpdir.chdir()
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(interface=LinearTestInterface(), name='mod1')
    mod2 = pe.MapNode(interface=LinearTestInterface(), iterfield=['input1'], name='mod2')
    pipe.connect([(mod1, mod2, [('output1', 'input1')])])
    pipe.base_dir = os.getcwd()
    mod1.inputs.input1 = 1
    execgraph = pipe.run(plugin='Linear')
    names = ['.'.join((node._hierarchy, node.name)) for node in execgraph.nodes()]
    node = list(execgraph.nodes())[names.index('pipe.mod1')]
    result = node.get_output('output1')
    assert result == [1, 1]