import os
import nipype.interfaces.spm.model as spm
import nipype.interfaces.matlab as mlab
def test_twosamplettestdesign():
    assert spm.TwoSampleTTestDesign._jobtype == 'stats'
    assert spm.TwoSampleTTestDesign._jobname == 'factorial_design'