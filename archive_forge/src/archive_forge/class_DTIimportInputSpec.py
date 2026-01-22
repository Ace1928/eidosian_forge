from nipype.interfaces.base import (
import os
class DTIimportInputSpec(CommandLineInputSpec):
    inputFile = File(position=-2, desc='Input DTI file', exists=True, argstr='%s')
    outputTensor = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output DTI volume', argstr='%s')
    testingmode = traits.Bool(desc='Enable testing mode. Sample helix file (helix-DTI.nhdr) will be loaded into Slicer and converted in Nifti.', argstr='--testingmode ')