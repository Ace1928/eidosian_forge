import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class MRISPreproc(FSCommand):
    """Use FreeSurfer mris_preproc to prepare a group of contrasts for
    a second level analysis

    Examples
    --------
    >>> preproc = MRISPreproc()
    >>> preproc.inputs.target = 'fsaverage'
    >>> preproc.inputs.hemi = 'lh'
    >>> preproc.inputs.vol_measure_file = [('cont1.nii', 'register.dat'),                                            ('cont1a.nii', 'register.dat')]
    >>> preproc.inputs.out_file = 'concatenated_file.mgz'
    >>> preproc.cmdline
    'mris_preproc --hemi lh --out concatenated_file.mgz --target fsaverage --iv cont1.nii register.dat --iv cont1a.nii register.dat'

    """
    _cmd = 'mris_preproc'
    input_spec = MRISPreprocInputSpec
    output_spec = MRISPreprocOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outfile = self.inputs.out_file
        outputs['out_file'] = outfile
        if not isdefined(outfile):
            outputs['out_file'] = os.path.join(os.getcwd(), 'concat_%s_%s.mgz' % (self.inputs.hemi, self.inputs.target))
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None