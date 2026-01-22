import os
from pathlib import Path
from nipype.interfaces.base import File, InputMultiPath, TraitedSpec, traits, isdefined
from nipype.interfaces.cat12.base import NestedCell, Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.utils.filemanip import split_filename
class ExtractAdditionalSurfaceParametersOutputSpec(TraitedSpec):
    lh_extracted_files = traits.List(File(exists=True), desc='Files of left Hemisphere extracted measures')
    rh_extracted_files = traits.List(File(exists=True), desc='Files of right Hemisphere extracted measures')
    lh_gyrification = traits.List(File(exists=True), desc='Gyrification of left Hemisphere')
    rh_gyrification = traits.List(File(exists=True), desc='Gyrification of right Hemisphere')
    lh_gmv = traits.List(File(exists=True), desc='Grey matter volume of left Hemisphere')
    rh_gmv = traits.List(File(exists=True), desc='Grey matter volume of right Hemisphere')
    lh_area = traits.List(File(exists=True), desc='Area of left Hemisphere')
    rh_area = traits.List(File(exists=True), desc='Area of right Hemisphere')
    lh_depth = traits.List(File(exists=True), desc='Depth of left Hemisphere')
    rh_depth = traits.List(File(exists=True), desc='Depth of right Hemisphere')
    lh_fractaldimension = traits.List(File(exists=True), desc='Fractal Dimension of left Hemisphere')
    rh_fractaldimension = traits.List(File(exists=True), desc='Fractal Dimension of right Hemisphere')