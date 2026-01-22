import os
from ...base import (
class BRAINSPosteriorToContinuousClassInputSpec(CommandLineInputSpec):
    inputWhiteVolume = File(desc='White Matter Posterior Volume', exists=True, argstr='--inputWhiteVolume %s')
    inputBasalGmVolume = File(desc='Basal Grey Matter Posterior Volume', exists=True, argstr='--inputBasalGmVolume %s')
    inputSurfaceGmVolume = File(desc='Surface Grey Matter Posterior Volume', exists=True, argstr='--inputSurfaceGmVolume %s')
    inputCsfVolume = File(desc='CSF Posterior Volume', exists=True, argstr='--inputCsfVolume %s')
    inputVbVolume = File(desc='Venous Blood Posterior Volume', exists=True, argstr='--inputVbVolume %s')
    inputCrblGmVolume = File(desc='Cerebellum Grey Matter Posterior Volume', exists=True, argstr='--inputCrblGmVolume %s')
    inputCrblWmVolume = File(desc='Cerebellum White Matter Posterior Volume', exists=True, argstr='--inputCrblWmVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output Continuous Tissue Classified Image', argstr='--outputVolume %s')