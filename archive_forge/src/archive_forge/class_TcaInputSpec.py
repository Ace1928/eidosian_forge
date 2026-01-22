import os
import re as regex
from ..base import (
class TcaInputSpec(CommandLineInputSpec):
    inputMaskFile = File(mandatory=True, desc='input mask volume', argstr='-i %s')
    outputMaskFile = File(desc='output mask volume. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    minCorrectionSize = traits.Int(2500, usedefault=True, desc='maximum correction size', argstr='-m %d')
    maxCorrectionSize = traits.Int(desc='minimum correction size', argstr='-n %d')
    foregroundDelta = traits.Int(20, usedefault=True, desc='foreground delta', argstr='--delta %d')
    verbosity = traits.Int(desc='verbosity (0 = quiet)', argstr='-v %d')
    timer = traits.Bool(desc='timing function', argstr='--timer')