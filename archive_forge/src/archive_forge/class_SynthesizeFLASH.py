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
class SynthesizeFLASH(FSCommand):
    """Synthesize a FLASH acquisition from T1 and proton density maps.

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import SynthesizeFLASH
    >>> syn = SynthesizeFLASH(tr=20, te=3, flip_angle=30)
    >>> syn.inputs.t1_image = 'T1.mgz'
    >>> syn.inputs.pd_image = 'PD.mgz'
    >>> syn.inputs.out_file = 'flash_30syn.mgz'
    >>> syn.cmdline
    'mri_synthesize 20.00 30.00 3.000 T1.mgz PD.mgz flash_30syn.mgz'

    """
    _cmd = 'mri_synthesize'
    input_spec = SynthesizeFLASHInputSpec
    output_spec = SynthesizeFLASHOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = self.inputs.out_file
        else:
            outputs['out_file'] = self._gen_fname('synth-flash_%02d.mgz' % self.inputs.flip_angle, suffix='')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['out_file']
        return None