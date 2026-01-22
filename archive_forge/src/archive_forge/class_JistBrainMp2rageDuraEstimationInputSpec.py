import os
from ..base import (
class JistBrainMp2rageDuraEstimationInputSpec(CommandLineInputSpec):
    inSecond = File(desc='Second inversion (Inv2) Image', exists=True, argstr='--inSecond %s')
    inSkull = File(desc='Skull Stripping Mask', exists=True, argstr='--inSkull %s')
    inDistance = traits.Float(desc='Distance to background (mm)', argstr='--inDistance %f')
    inoutput = traits.Enum('dura_region', 'boundary', 'dura_prior', 'bg_prior', 'intens_prior', desc='Outputs an estimate of the dura / CSF boundary or an estimate of the entire dura region.', argstr='--inoutput %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outDura = traits.Either(traits.Bool, File(), hash_files=False, desc='Dura Image', argstr='--outDura %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)