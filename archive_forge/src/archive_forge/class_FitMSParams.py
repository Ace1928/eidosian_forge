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
class FitMSParams(FSCommand):
    """Estimate tissue parameters from a set of FLASH images.

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import FitMSParams
    >>> msfit = FitMSParams()
    >>> msfit.inputs.in_files = ['flash_05.mgz', 'flash_30.mgz']
    >>> msfit.inputs.out_dir = 'flash_parameters'
    >>> msfit.cmdline
    'mri_ms_fitparms  flash_05.mgz flash_30.mgz flash_parameters'

    """
    _cmd = 'mri_ms_fitparms'
    input_spec = FitMSParamsInputSpec
    output_spec = FitMSParamsOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'in_files':
            cmd = ''
            for i, file in enumerate(value):
                if isdefined(self.inputs.tr_list):
                    cmd = ' '.join((cmd, '-tr %.1f' % self.inputs.tr_list[i]))
                if isdefined(self.inputs.te_list):
                    cmd = ' '.join((cmd, '-te %.3f' % self.inputs.te_list[i]))
                if isdefined(self.inputs.flip_list):
                    cmd = ' '.join((cmd, '-fa %.1f' % self.inputs.flip_list[i]))
                if isdefined(self.inputs.xfm_list):
                    cmd = ' '.join((cmd, '-at %s' % self.inputs.xfm_list[i]))
                cmd = ' '.join((cmd, file))
            return cmd
        return super(FitMSParams, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_dir):
            out_dir = self._gen_filename('out_dir')
        else:
            out_dir = self.inputs.out_dir
        outputs['t1_image'] = os.path.join(out_dir, 'T1.mgz')
        outputs['pd_image'] = os.path.join(out_dir, 'PD.mgz')
        outputs['t2star_image'] = os.path.join(out_dir, 'T2star.mgz')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_dir':
            return os.getcwd()
        return None