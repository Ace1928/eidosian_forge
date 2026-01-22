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
class MRIsCALabel(FSCommandOpenMP):
    """
    For a single subject, produces an annotation file, in which each
    cortical surface vertex is assigned a neuroanatomical label.This
    automatic procedure employs data from a previously-prepared atlas
    file. An atlas file is created from a training set, capturing region
    data manually drawn by neuroanatomists combined with statistics on
    variability correlated to geometric information derived from the
    cortical model (sulcus and curvature). Besides the atlases provided
    with FreeSurfer, new ones can be prepared using mris_ca_train).

    Examples
    ========

    >>> from nipype.interfaces import freesurfer
    >>> ca_label = freesurfer.MRIsCALabel()
    >>> ca_label.inputs.subject_id = "test"
    >>> ca_label.inputs.hemisphere = "lh"
    >>> ca_label.inputs.canonsurf = "lh.pial"
    >>> ca_label.inputs.curv = "lh.pial"
    >>> ca_label.inputs.sulc = "lh.pial"
    >>> ca_label.inputs.classifier = "im1.nii" # in pracice, use .gcs extension
    >>> ca_label.inputs.smoothwm = "lh.pial"
    >>> ca_label.cmdline
    'mris_ca_label test lh lh.pial im1.nii lh.aparc.annot'
    """
    _cmd = 'mris_ca_label'
    input_spec = MRIsCALabelInputSpec
    output_spec = MRIsCALabelOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.canonsurf, folder='surf')
            copy2subjdir(self, self.inputs.smoothwm, folder='surf', basename='{0}.smoothwm'.format(self.inputs.hemisphere))
            copy2subjdir(self, self.inputs.curv, folder='surf', basename='{0}.curv'.format(self.inputs.hemisphere))
            copy2subjdir(self, self.inputs.sulc, folder='surf', basename='{0}.sulc'.format(self.inputs.hemisphere))
        label_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label')
        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)
        return super(MRIsCALabel, self).run(**inputs)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_basename = os.path.basename(self.inputs.out_file)
        outputs['out_file'] = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label', out_basename)
        return outputs