import numpy as np
import scipy.sparse as ssp
import re
from unittest import mock
from nipype.pipeline.plugins.tools import report_crash
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

Can use the following code to test that a mapnode crash continues successfully
Need to put this into a unit-test with a timeout

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

wf = pe.Workflow(name='test')

def func(arg1):
    if arg1 == 2:
        raise Exception('arg cannot be ' + str(arg1))
    return arg1

funkynode = pe.MapNode(niu.Function(function=func, input_names=['arg1'],
                                                   output_names=['out']),
                       iterfield=['arg1'],
                       name = 'functor')
funkynode.inputs.arg1 = [1,2]

wf.add_nodes([funkynode])
wf.base_dir = '/tmp'

wf.run(plugin='MultiProc')
