import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class MRICoregInputSpec(FSTraitedSpec):
    source_file = File(argstr='--mov %s', desc='source file to be registered', mandatory=True, copyfile=False)
    reference_file = File(argstr='--ref %s', desc='reference (target) file', mandatory=True, copyfile=False, xor=['subject_id'])
    out_lta_file = traits.Either(True, File, argstr='--lta %s', default=True, usedefault=True, desc='output registration file (LTA format)')
    out_reg_file = traits.Either(True, File, argstr='--regdat %s', desc='output registration file (REG format)')
    out_params_file = traits.Either(True, File, argstr='--params %s', desc='output parameters file')
    subjects_dir = Directory(exists=True, argstr='--sd %s', desc='FreeSurfer SUBJECTS_DIR')
    subject_id = traits.Str(argstr='--s %s', position=1, mandatory=True, xor=['reference_file'], requires=['subjects_dir'], desc='freesurfer subject ID (implies ``reference_mask == aparc+aseg.mgz`` unless otherwise specified)')
    dof = traits.Enum(6, 9, 12, argstr='--dof %d', desc='number of transform degrees of freedom')
    reference_mask = traits.Either(False, traits.Str, argstr='--ref-mask %s', position=2, desc='mask reference volume with given mask, or None if ``False``')
    source_mask = traits.Str(argstr='--mov-mask', desc='mask source file with given mask')
    num_threads = traits.Int(argstr='--threads %d', desc='number of OpenMP threads')
    no_coord_dithering = traits.Bool(argstr='--no-coord-dither', desc='turn off coordinate dithering')
    no_intensity_dithering = traits.Bool(argstr='--no-intensity-dither', desc='turn off intensity dithering')
    sep = traits.List(argstr='--sep %s...', minlen=1, maxlen=2, desc='set spatial scales, in voxels (default [2, 4])')
    initial_translation = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--trans %g %g %g', desc='initial translation in mm (implies no_cras0)')
    initial_rotation = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--rot %g %g %g', desc='initial rotation in degrees')
    initial_scale = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--scale %g %g %g', desc='initial scale')
    initial_shear = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--shear %g %g %g', desc='initial shear (Hxy, Hxz, Hyz)')
    no_cras0 = traits.Bool(argstr='--no-cras0', desc='do not set translation parameters to align centers of source and reference files')
    max_iters = traits.Range(low=1, argstr='--nitersmax %d', desc='maximum iterations (default: 4)')
    ftol = traits.Float(argstr='--ftol %e', desc='floating-point tolerance (default=1e-7)')
    linmintol = traits.Float(argstr='--linmintol %e')
    saturation_threshold = traits.Range(low=0.0, high=100.0, argstr='--sat %g', desc='saturation threshold (default=9.999)')
    conform_reference = traits.Bool(argstr='--conf-ref', desc='conform reference without rescaling')
    no_brute_force = traits.Bool(argstr='--no-bf', desc='do not brute force search')
    brute_force_limit = traits.Float(argstr='--bf-lim %g', xor=['no_brute_force'], desc='constrain brute force search to +/- lim')
    brute_force_samples = traits.Int(argstr='--bf-nsamp %d', xor=['no_brute_force'], desc='number of samples in brute force search')
    no_smooth = traits.Bool(argstr='--no-smooth', desc='do not apply smoothing to either reference or source file')
    ref_fwhm = traits.Float(argstr='--ref-fwhm', desc='apply smoothing to reference file')
    source_oob = traits.Bool(argstr='--mov-oob', desc='count source voxels that are out-of-bounds as 0')