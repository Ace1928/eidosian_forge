import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Aparc2AsegInputSpec(FSTraitedSpec):
    subject_id = traits.String('subject_id', argstr='--s %s', usedefault=True, mandatory=True, desc='Subject being processed')
    out_file = File(argstr='--o %s', exists=False, mandatory=True, desc='Full path of file to save the output segmentation in')
    lh_white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/lh.white')
    rh_white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/rh.white')
    lh_pial = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/lh.pial')
    rh_pial = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/rh.pial')
    lh_ribbon = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/lh.ribbon.mgz')
    rh_ribbon = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/rh.ribbon.mgz')
    ribbon = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/ribbon.mgz')
    lh_annotation = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/label/lh.aparc.annot')
    rh_annotation = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/label/rh.aparc.annot')
    filled = File(exists=True, desc='Implicit input filled file. Only required with FS v5.3.')
    aseg = File(argstr='--aseg %s', exists=True, desc='Input aseg file')
    volmask = traits.Bool(argstr='--volmask', desc='Volume mask flag')
    ctxseg = File(argstr='--ctxseg %s', exists=True, desc='')
    label_wm = traits.Bool(argstr='--labelwm', desc='For each voxel labeled as white matter in the aseg, re-assign\nits label to be that of the closest cortical point if its\ndistance is less than dmaxctx.')
    hypo_wm = traits.Bool(argstr='--hypo-as-wm', desc='Label hypointensities as WM')
    rip_unknown = traits.Bool(argstr='--rip-unknown', desc="Do not label WM based on 'unknown' corical label")
    a2009s = traits.Bool(argstr='--a2009s', desc='Using the a2009s atlas')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.This will copy the input files to the node directory.')