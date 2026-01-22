import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class KellyKapowskiOutputSpec(TraitedSpec):
    cortical_thickness = File(desc='A thickness map defined in the segmented gray matter.')
    warped_white_matter = File(desc='A warped white matter image.')