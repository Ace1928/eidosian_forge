import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class MRISPreprocReconAll(MRISPreproc):
    """Extends MRISPreproc to allow it to be used in a recon-all workflow

    Examples
    --------
    >>> preproc = MRISPreprocReconAll()
    >>> preproc.inputs.target = 'fsaverage'
    >>> preproc.inputs.hemi = 'lh'
    >>> preproc.inputs.vol_measure_file = [('cont1.nii', 'register.dat'),                                            ('cont1a.nii', 'register.dat')]
    >>> preproc.inputs.out_file = 'concatenated_file.mgz'
    >>> preproc.cmdline
    'mris_preproc --hemi lh --out concatenated_file.mgz --s subject_id --target fsaverage --iv cont1.nii register.dat --iv cont1a.nii register.dat'

    """
    input_spec = MRISPreprocReconAllInputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            if isdefined(self.inputs.surf_dir):
                folder = self.inputs.surf_dir
            else:
                folder = 'surf'
            if isdefined(self.inputs.surfreg_files):
                for surfreg in self.inputs.surfreg_files:
                    basename = os.path.basename(surfreg)
                    copy2subjdir(self, surfreg, folder, basename)
                    if basename.startswith('lh.'):
                        copy2subjdir(self, self.inputs.lh_surfreg_target, folder, basename, subject_id=self.inputs.target)
                    else:
                        copy2subjdir(self, self.inputs.rh_surfreg_target, folder, basename, subject_id=self.inputs.target)
            if isdefined(self.inputs.surf_measure_file):
                copy2subjdir(self, self.inputs.surf_measure_file, folder)
        return super(MRISPreprocReconAll, self).run(**inputs)

    def _format_arg(self, name, spec, value):
        if name == 'surfreg_files':
            basename = os.path.basename(value[0])
            return spec.argstr % basename.lstrip('rh.').lstrip('lh.')
        if name == 'surf_measure_file':
            basename = os.path.basename(value)
            return spec.argstr % basename.lstrip('rh.').lstrip('lh.')
        return super(MRISPreprocReconAll, self)._format_arg(name, spec, value)