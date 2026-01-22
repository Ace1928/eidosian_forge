import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class ReconAllInputSpec(CommandLineInputSpec):
    subject_id = traits.Str('recon_all', argstr='-subjid %s', desc='subject name', usedefault=True)
    directive = traits.Enum('all', 'autorecon1', 'autorecon2', 'autorecon2-volonly', 'autorecon2-perhemi', 'autorecon2-inflate1', 'autorecon2-cp', 'autorecon2-wm', 'autorecon3', 'autorecon3-T2pial', 'autorecon-pial', 'autorecon-hemi', 'localGI', 'qcache', argstr='-%s', desc='process directive', usedefault=True, position=0)
    hemi = traits.Enum('lh', 'rh', desc='hemisphere to process', argstr='-hemi %s')
    T1_files = InputMultiPath(File(exists=True), argstr='-i %s...', desc='name of T1 file to process')
    T2_file = File(exists=True, argstr='-T2 %s', min_ver='5.3.0', desc='Convert T2 image to orig directory')
    FLAIR_file = File(exists=True, argstr='-FLAIR %s', min_ver='5.3.0', desc='Convert FLAIR image to orig directory')
    use_T2 = traits.Bool(argstr='-T2pial', min_ver='5.3.0', xor=['use_FLAIR'], desc='Use T2 image to refine the pial surface')
    use_FLAIR = traits.Bool(argstr='-FLAIRpial', min_ver='5.3.0', xor=['use_T2'], desc='Use FLAIR image to refine the pial surface')
    openmp = traits.Int(argstr='-openmp %d', desc='Number of processors to use in parallel')
    parallel = traits.Bool(argstr='-parallel', desc='Enable parallel execution')
    hires = traits.Bool(argstr='-hires', min_ver='6.0.0', desc='Conform to minimum voxel size (for voxels < 1mm)')
    mprage = traits.Bool(argstr='-mprage', desc='Assume scan parameters are MGH MP-RAGE protocol, which produces darker gray matter')
    big_ventricles = traits.Bool(argstr='-bigventricles', desc='For use in subjects with enlarged ventricles')
    brainstem = traits.Bool(argstr='-brainstem-structures', desc='Segment brainstem structures')
    hippocampal_subfields_T1 = traits.Bool(argstr='-hippocampal-subfields-T1', min_ver='6.0.0', desc='segment hippocampal subfields using input T1 scan')
    hippocampal_subfields_T2 = traits.Tuple(File(exists=True), traits.Str(), argstr='-hippocampal-subfields-T2 %s %s', min_ver='6.0.0', desc='segment hippocampal subfields using T2 scan, identified by ID (may be combined with hippocampal_subfields_T1)')
    expert = File(exists=True, argstr='-expert %s', desc='Set parameters using expert file')
    xopts = traits.Enum('use', 'clean', 'overwrite', argstr='-xopts-%s', desc='Use, delete or overwrite existing expert options file')
    subjects_dir = Directory(exists=True, argstr='-sd %s', hash_files=False, desc='path to subjects directory', genfile=True)
    flags = InputMultiPath(traits.Str, argstr='%s', desc='additional parameters')
    talairach = traits.Str(desc='Flags to pass to talairach commands', xor=['expert'])
    mri_normalize = traits.Str(desc='Flags to pass to mri_normalize commands', xor=['expert'])
    mri_watershed = traits.Str(desc='Flags to pass to mri_watershed commands', xor=['expert'])
    mri_em_register = traits.Str(desc='Flags to pass to mri_em_register commands', xor=['expert'])
    mri_ca_normalize = traits.Str(desc='Flags to pass to mri_ca_normalize commands', xor=['expert'])
    mri_ca_register = traits.Str(desc='Flags to pass to mri_ca_register commands', xor=['expert'])
    mri_remove_neck = traits.Str(desc='Flags to pass to mri_remove_neck commands', xor=['expert'])
    mri_ca_label = traits.Str(desc='Flags to pass to mri_ca_label commands', xor=['expert'])
    mri_segstats = traits.Str(desc='Flags to pass to mri_segstats commands', xor=['expert'])
    mri_mask = traits.Str(desc='Flags to pass to mri_mask commands', xor=['expert'])
    mri_segment = traits.Str(desc='Flags to pass to mri_segment commands', xor=['expert'])
    mri_edit_wm_with_aseg = traits.Str(desc='Flags to pass to mri_edit_wm_with_aseg commands', xor=['expert'])
    mri_pretess = traits.Str(desc='Flags to pass to mri_pretess commands', xor=['expert'])
    mri_fill = traits.Str(desc='Flags to pass to mri_fill commands', xor=['expert'])
    mri_tessellate = traits.Str(desc='Flags to pass to mri_tessellate commands', xor=['expert'])
    mris_smooth = traits.Str(desc='Flags to pass to mri_smooth commands', xor=['expert'])
    mris_inflate = traits.Str(desc='Flags to pass to mri_inflate commands', xor=['expert'])
    mris_sphere = traits.Str(desc='Flags to pass to mris_sphere commands', xor=['expert'])
    mris_fix_topology = traits.Str(desc='Flags to pass to mris_fix_topology commands', xor=['expert'])
    mris_make_surfaces = traits.Str(desc='Flags to pass to mris_make_surfaces commands', xor=['expert'])
    mris_surf2vol = traits.Str(desc='Flags to pass to mris_surf2vol commands', xor=['expert'])
    mris_register = traits.Str(desc='Flags to pass to mris_register commands', xor=['expert'])
    mrisp_paint = traits.Str(desc='Flags to pass to mrisp_paint commands', xor=['expert'])
    mris_ca_label = traits.Str(desc='Flags to pass to mris_ca_label commands', xor=['expert'])
    mris_anatomical_stats = traits.Str(desc='Flags to pass to mris_anatomical_stats commands', xor=['expert'])
    mri_aparc2aseg = traits.Str(desc='Flags to pass to mri_aparc2aseg commands', xor=['expert'])