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
class SegmentCC(FSCommand):
    """
    This program segments the corpus callosum into five separate labels in
    the subcortical segmentation volume 'aseg.mgz'. The divisions of the
    cc are equally spaced in terms of distance along the primary
    eigendirection (pretty much the long axis) of the cc. The lateral
    extent can be changed with the -T <thickness> parameter, where
    <thickness> is the distance off the midline (so -T 1 would result in
    the who CC being 3mm thick). The default is 2 so it's 5mm thick. The
    aseg.stats values should be volume.

    Examples
    ========
    >>> from nipype.interfaces import freesurfer
    >>> SegmentCC_node = freesurfer.SegmentCC()
    >>> SegmentCC_node.inputs.in_file = "aseg.mgz"
    >>> SegmentCC_node.inputs.in_norm = "norm.mgz"
    >>> SegmentCC_node.inputs.out_rotation = "cc.lta"
    >>> SegmentCC_node.inputs.subject_id = "test"
    >>> SegmentCC_node.cmdline
    'mri_cc -aseg aseg.mgz -o aseg.auto.mgz -lta cc.lta test'
    """
    _cmd = 'mri_cc'
    input_spec = SegmentCCInputSpec
    output_spec = SegmentCCOutputSpec

    def _format_arg(self, name, spec, value):
        if name in ['in_file', 'in_norm', 'out_file']:
            basename = os.path.basename(value)
            return spec.argstr % basename
        return super(SegmentCC, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        outputs['out_rotation'] = os.path.abspath(self.inputs.out_rotation)
        return outputs

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            for originalfile in [self.inputs.in_file, self.inputs.in_norm]:
                copy2subjdir(self, originalfile, folder='mri')
        return super(SegmentCC, self).run(**inputs)

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        predicted_outputs = self._list_outputs()
        for name in ['out_file', 'out_rotation']:
            out_file = predicted_outputs[name]
            if not os.path.isfile(out_file):
                out_base = os.path.basename(out_file)
                if isdefined(self.inputs.subjects_dir):
                    subj_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id)
                else:
                    subj_dir = os.path.join(os.getcwd(), self.inputs.subject_id)
                if name == 'out_file':
                    out_tmp = os.path.join(subj_dir, 'mri', out_base)
                elif name == 'out_rotation':
                    out_tmp = os.path.join(subj_dir, 'mri', 'transforms', out_base)
                else:
                    out_tmp = None
                if out_tmp and os.path.isfile(out_tmp):
                    if not os.path.isdir(os.path.dirname(out_tmp)):
                        os.makedirs(os.path.dirname(out_tmp))
                    shutil.move(out_tmp, out_file)
        return super(SegmentCC, self).aggregate_outputs(runtime, needed_outputs)