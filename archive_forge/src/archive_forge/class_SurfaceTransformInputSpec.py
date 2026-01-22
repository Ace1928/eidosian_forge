import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceTransformInputSpec(FSTraitedSpec):
    source_file = File(exists=True, mandatory=True, argstr='--sval %s', xor=['source_annot_file'], desc='surface file with source values')
    source_annot_file = File(exists=True, mandatory=True, argstr='--sval-annot %s', xor=['source_file'], desc='surface annotation file')
    source_subject = traits.String(mandatory=True, argstr='--srcsubject %s', desc='subject id for source surface')
    hemi = traits.Enum('lh', 'rh', argstr='--hemi %s', mandatory=True, desc='hemisphere to transform')
    target_subject = traits.String(mandatory=True, argstr='--trgsubject %s', desc='subject id of target surface')
    target_ico_order = traits.Enum(1, 2, 3, 4, 5, 6, 7, argstr='--trgicoorder %d', desc="order of the icosahedron if target_subject is 'ico'")
    source_type = traits.Enum(filetypes, argstr='--sfmt %s', requires=['source_file'], desc='source file format')
    target_type = traits.Enum(filetypes + implicit_filetypes, argstr='--tfmt %s', desc='output format')
    reshape = traits.Bool(argstr='--reshape', desc='reshape output surface to conform with Nifti')
    reshape_factor = traits.Int(argstr='--reshape-factor', desc='number of slices in reshaped image')
    out_file = File(argstr='--tval %s', genfile=True, desc='surface file to write')