import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class BrainExtractionOutputSpec(TraitedSpec):
    BrainExtractionMask = File(exists=True, desc='brain extraction mask')
    BrainExtractionBrain = File(exists=True, desc='brain extraction image')
    BrainExtractionCSF = File(exists=True, desc='segmentation mask with only CSF')
    BrainExtractionGM = File(exists=True, desc='segmentation mask with only grey matter')
    BrainExtractionInitialAffine = File(exists=True, desc='')
    BrainExtractionInitialAffineFixed = File(exists=True, desc='')
    BrainExtractionInitialAffineMoving = File(exists=True, desc='')
    BrainExtractionLaplacian = File(exists=True, desc='')
    BrainExtractionPrior0GenericAffine = File(exists=True, desc='')
    BrainExtractionPrior1InverseWarp = File(exists=True, desc='')
    BrainExtractionPrior1Warp = File(exists=True, desc='')
    BrainExtractionPriorWarped = File(exists=True, desc='')
    BrainExtractionSegmentation = File(exists=True, desc='segmentation mask with CSF, GM, and WM')
    BrainExtractionTemplateLaplacian = File(exists=True, desc='')
    BrainExtractionTmp = File(exists=True, desc='')
    BrainExtractionWM = File(exists=True, desc='segmenration mask with only white matter')
    N4Corrected0 = File(exists=True, desc='N4 bias field corrected image')
    N4Truncated0 = File(exists=True, desc='')