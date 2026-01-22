import os
from ...base import (
class extractNrrdVectorIndexInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input file containing the vector that will be extracted', exists=True, argstr='--inputVolume %s')
    vectorIndex = traits.Int(desc='Index in the vector image to extract', argstr='--vectorIndex %d')
    setImageOrientation = traits.Enum('AsAcquired', 'Axial', 'Coronal', 'Sagittal', desc='Sets the image orientation of the extracted vector (Axial, Coronal, Sagittal)', argstr='--setImageOrientation %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the vector image at the given index', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')