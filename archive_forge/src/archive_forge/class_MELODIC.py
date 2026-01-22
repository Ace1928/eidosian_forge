import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MELODIC(FSLCommand):
    """Multivariate Exploratory Linear Optimised Decomposition into Independent
    Components

    Examples
    --------

    >>> melodic_setup = MELODIC()
    >>> melodic_setup.inputs.approach = 'tica'
    >>> melodic_setup.inputs.in_files = ['functional.nii', 'functional2.nii', 'functional3.nii']
    >>> melodic_setup.inputs.no_bet = True
    >>> melodic_setup.inputs.bg_threshold = 10
    >>> melodic_setup.inputs.tr_sec = 1.5
    >>> melodic_setup.inputs.mm_thresh = 0.5
    >>> melodic_setup.inputs.out_stats = True
    >>> melodic_setup.inputs.t_des = 'timeDesign.mat'
    >>> melodic_setup.inputs.t_con = 'timeDesign.con'
    >>> melodic_setup.inputs.s_des = 'subjectDesign.mat'
    >>> melodic_setup.inputs.s_con = 'subjectDesign.con'
    >>> melodic_setup.inputs.out_dir = 'groupICA.out'
    >>> melodic_setup.cmdline
    'melodic -i functional.nii,functional2.nii,functional3.nii -a tica --bgthreshold=10.000000 --mmthresh=0.500000 --nobet -o groupICA.out --Ostats --Scon=subjectDesign.con --Sdes=subjectDesign.mat --Tcon=timeDesign.con --Tdes=timeDesign.mat --tr=1.500000'
    >>> melodic_setup.run() # doctest: +SKIP


    """
    input_spec = MELODICInputSpec
    output_spec = MELODICOutputSpec
    _cmd = 'melodic'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_dir):
            outputs['out_dir'] = os.path.abspath(self.inputs.out_dir)
        else:
            outputs['out_dir'] = self._gen_filename('out_dir')
        if isdefined(self.inputs.report) and self.inputs.report:
            outputs['report_dir'] = os.path.join(outputs['out_dir'], 'report')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_dir':
            return os.getcwd()