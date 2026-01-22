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
class FEAT(FSLCommand):
    """Uses FSL feat to calculate first level stats"""
    _cmd = 'feat'
    input_spec = FEATInputSpec
    output_spec = FEATOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        is_ica = False
        outputs['feat_dir'] = None
        with open(self.inputs.fsf_file, 'rt') as fp:
            text = fp.read()
            if 'set fmri(inmelodic) 1' in text:
                is_ica = True
            for line in text.split('\n'):
                if line.find('set fmri(outputdir)') > -1:
                    try:
                        outputdir_spec = line.split('"')[-2]
                        if os.path.exists(outputdir_spec):
                            outputs['feat_dir'] = outputdir_spec
                    except:
                        pass
        if not outputs['feat_dir']:
            if is_ica:
                outputs['feat_dir'] = glob(os.path.join(os.getcwd(), '*ica'))[0]
            else:
                outputs['feat_dir'] = glob(os.path.join(os.getcwd(), '*feat'))[0]
        return outputs