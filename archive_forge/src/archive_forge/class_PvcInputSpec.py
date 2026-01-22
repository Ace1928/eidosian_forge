import os
import re as regex
from ..base import (
class PvcInputSpec(CommandLineInputSpec):
    inputMRIFile = File(mandatory=True, desc='MRI file', argstr='-i %s')
    inputMaskFile = File(desc='brain mask file', argstr='-m %s')
    outputLabelFile = File(desc='output label file. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    outputTissueFractionFile = File(desc='output tissue fraction file', argstr='-f %s', genfile=True)
    spatialPrior = traits.Float(desc='spatial prior strength', argstr='-l %f')
    verbosity = traits.Int(desc='verbosity level (0 = silent)', argstr='-v %d')
    threeClassFlag = traits.Bool(desc='use a three-class (CSF=0,GM=1,WM=2) labeling', argstr='-3')
    timer = traits.Bool(desc='time processing', argstr='--timer')