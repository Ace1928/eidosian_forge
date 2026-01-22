import os
from ..base import (
class DWICompareInputSpec(CommandLineInputSpec):
    inputVolume1 = File(desc='First input volume (.nhdr or .nrrd)', exists=True, argstr='--inputVolume1 %s')
    inputVolume2 = File(desc='Second input volume (.nhdr or .nrrd)', exists=True, argstr='--inputVolume2 %s')