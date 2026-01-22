import os
from ... import logging
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from .base import FSCommand, FSTraitedSpec, FSCommandOpenMP, FSTraitedSpecOpenMP
class RobustTemplateInputSpec(FSTraitedSpecOpenMP):
    in_files = InputMultiPath(File(exists=True), mandatory=True, argstr='--mov %s', desc='input movable volumes to be aligned to common mean/median template')
    out_file = File('mri_robust_template_out.mgz', mandatory=True, usedefault=True, argstr='--template %s', desc='output template volume (final mean/median image)')
    auto_detect_sensitivity = traits.Bool(argstr='--satit', xor=['outlier_sensitivity'], mandatory=True, desc='auto-detect good sensitivity (recommended for head or full brain scans)')
    outlier_sensitivity = traits.Float(argstr='--sat %.4f', xor=['auto_detect_sensitivity'], mandatory=True, desc='set outlier sensitivity manually (e.g. "--sat 4.685" ). Higher values mean less sensitivity.')
    transform_outputs = traits.Either(InputMultiPath(File(exists=False)), traits.Bool, argstr='--lta %s', desc='output xforms to template (for each input)')
    intensity_scaling = traits.Bool(default_value=False, argstr='--iscale', desc='allow also intensity scaling (default off)')
    scaled_intensity_outputs = traits.Either(InputMultiPath(File(exists=False)), traits.Bool, argstr='--iscaleout %s', desc='final intensity scales (will activate --iscale)')
    subsample_threshold = traits.Int(argstr='--subsample %d', desc='subsample if dim > # on all axes (default no subs.)')
    average_metric = traits.Enum('median', 'mean', argstr='--average %d', desc='construct template from: 0 Mean, 1 Median (default)')
    initial_timepoint = traits.Int(argstr='--inittp %d', desc='use TP# for spacial init (default random), 0: no init')
    fixed_timepoint = traits.Bool(default_value=False, argstr='--fixtp', desc='map everything to init TP# (init TP is not resampled)')
    no_iteration = traits.Bool(default_value=False, argstr='--noit', desc='do not iterate, just create first template')
    initial_transforms = InputMultiPath(File(exists=True), argstr='--ixforms %s', desc='use initial transforms (lta) on source')
    in_intensity_scales = InputMultiPath(File(exists=True), argstr='--iscalein %s', desc='use initial intensity scales')