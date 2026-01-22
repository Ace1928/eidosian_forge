import os
from ... import logging
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from .base import FSCommand, FSTraitedSpec, FSCommandOpenMP, FSTraitedSpecOpenMP
class FuseSegmentationsOutputSpec(TraitedSpec):
    out_file = File(exists=False, desc='output fused segmentation file')