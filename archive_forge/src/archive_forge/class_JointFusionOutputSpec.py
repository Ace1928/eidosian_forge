import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class JointFusionOutputSpec(TraitedSpec):
    out_label_fusion = File(exists=True)
    out_intensity_fusion = OutputMultiPath(File(exists=True))
    out_label_post_prob = OutputMultiPath(File(exists=True))
    out_atlas_voting_weight = OutputMultiPath(File(exists=True))