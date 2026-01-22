import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class MRISPreprocInputSpec(FSTraitedSpec):
    out_file = File(argstr='--out %s', genfile=True, desc='output filename')
    target = traits.Str(argstr='--target %s', mandatory=True, desc='target subject name')
    hemi = traits.Enum('lh', 'rh', argstr='--hemi %s', mandatory=True, desc='hemisphere for source and target')
    surf_measure = traits.Str(argstr='--meas %s', xor=('surf_measure', 'surf_measure_file', 'surf_area'), desc='Use subject/surf/hemi.surf_measure as input')
    surf_area = traits.Str(argstr='--area %s', xor=('surf_measure', 'surf_measure_file', 'surf_area'), desc='Extract vertex area from subject/surf/hemi.surfname to use as input.')
    subjects = traits.List(argstr='--s %s...', xor=('subjects', 'fsgd_file', 'subject_file'), desc='subjects from who measures are calculated')
    fsgd_file = File(exists=True, argstr='--fsgd %s', xor=('subjects', 'fsgd_file', 'subject_file'), desc='specify subjects using fsgd file')
    subject_file = File(exists=True, argstr='--f %s', xor=('subjects', 'fsgd_file', 'subject_file'), desc='file specifying subjects separated by white space')
    surf_measure_file = InputMultiPath(File(exists=True), argstr='--is %s...', xor=('surf_measure', 'surf_measure_file', 'surf_area'), desc='file alternative to surfmeas, still requires list of subjects')
    source_format = traits.Str(argstr='--srcfmt %s', desc='source format')
    surf_dir = traits.Str(argstr='--surfdir %s', desc='alternative directory (instead of surf)')
    vol_measure_file = InputMultiPath(traits.Tuple(File(exists=True), File(exists=True)), argstr='--iv %s %s...', desc='list of volume measure and reg file tuples')
    proj_frac = traits.Float(argstr='--projfrac %s', desc='projection fraction for vol2surf')
    fwhm = traits.Float(argstr='--fwhm %f', xor=['num_iters'], desc='smooth by fwhm mm on the target surface')
    num_iters = traits.Int(argstr='--niters %d', xor=['fwhm'], desc='niters : smooth by niters on the target surface')
    fwhm_source = traits.Float(argstr='--fwhm-src %f', xor=['num_iters_source'], desc='smooth by fwhm mm on the source surface')
    num_iters_source = traits.Int(argstr='--niterssrc %d', xor=['fwhm_source'], desc='niters : smooth by niters on the source surface')
    smooth_cortex_only = traits.Bool(argstr='--smooth-cortex-only', desc='only smooth cortex (ie, exclude medial wall)')