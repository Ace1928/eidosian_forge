import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
class CalcTopNCC(NiftySegCommand):
    """Interface for executable seg_CalcTopNCC from NiftySeg platform.

    Examples
    --------
    >>> from nipype.interfaces import niftyseg
    >>> node = niftyseg.CalcTopNCC()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.num_templates = 2
    >>> node.inputs.in_templates = ['im2.nii', 'im3.nii']
    >>> node.inputs.top_templates = 1
    >>> node.cmdline
    'seg_CalcTopNCC -target im1.nii -templates 2 im2.nii im3.nii -n 1'

    """
    _cmd = get_custom_path('seg_CalcTopNCC', env_dir='NIFTYSEGDIR')
    _suffix = '_topNCC'
    input_spec = CalcTopNCCInputSpec
    output_spec = CalcTopNCCOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        outfile = os.path.join(os.getcwd(), 'CalcTopNCC.json')
        if runtime is None or not runtime.stdout:
            try:
                out_files = load_json(outfile)['files']
            except IOError:
                return self.run().outputs
        else:
            out_files = []
            for line in runtime.stdout.split('\n'):
                if line:
                    values = line.split()
                    if len(values) > 1:
                        out_files.append([str(val) for val in values])
                    else:
                        out_files.extend([str(val) for val in values])
            if len(out_files) == 1:
                out_files = out_files[0]
            save_json(outfile, dict(files=out_files))
        outputs.out_files = out_files
        return outputs