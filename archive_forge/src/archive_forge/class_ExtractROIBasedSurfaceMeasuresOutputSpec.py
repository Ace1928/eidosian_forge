import os
from pathlib import Path
from nipype.interfaces.base import File, InputMultiPath, TraitedSpec, traits, isdefined
from nipype.interfaces.cat12.base import NestedCell, Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.utils.filemanip import split_filename
class ExtractROIBasedSurfaceMeasuresOutputSpec(TraitedSpec):
    label_files = traits.List(File(exists=True), desc='Files with the measures extracted for ROIs.')