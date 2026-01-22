import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProbTrackX2(ProbTrackX):
    """Use FSL  probtrackx2 for tractography on bedpostx results

    Examples
    --------

    >>> from nipype.interfaces import fsl
    >>> pbx2 = fsl.ProbTrackX2()
    >>> pbx2.inputs.seed = 'seed_source.nii.gz'
    >>> pbx2.inputs.thsamples = 'merged_th1samples.nii.gz'
    >>> pbx2.inputs.fsamples = 'merged_f1samples.nii.gz'
    >>> pbx2.inputs.phsamples = 'merged_ph1samples.nii.gz'
    >>> pbx2.inputs.mask = 'nodif_brain_mask.nii.gz'
    >>> pbx2.inputs.out_dir = '.'
    >>> pbx2.inputs.n_samples = 3
    >>> pbx2.inputs.n_steps = 10
    >>> pbx2.cmdline
    'probtrackx2 --forcedir -m nodif_brain_mask.nii.gz --nsamples=3 --nsteps=10 --opd --dir=. --samples=merged --seed=seed_source.nii.gz'
    """
    _cmd = 'probtrackx2'
    input_spec = ProbTrackX2InputSpec
    output_spec = ProbTrackX2OutputSpec

    def _list_outputs(self):
        outputs = super(ProbTrackX2, self)._list_outputs()
        if not isdefined(self.inputs.out_dir):
            out_dir = os.getcwd()
        else:
            out_dir = self.inputs.out_dir
        outputs['way_total'] = os.path.abspath(os.path.join(out_dir, 'waytotal'))
        if isdefined(self.inputs.omatrix1):
            outputs['network_matrix'] = os.path.abspath(os.path.join(out_dir, 'matrix_seeds_to_all_targets'))
            outputs['matrix1_dot'] = os.path.abspath(os.path.join(out_dir, 'fdt_matrix1.dot'))
        if isdefined(self.inputs.omatrix2):
            outputs['lookup_tractspace'] = os.path.abspath(os.path.join(out_dir, 'lookup_tractspace_fdt_matrix2.nii.gz'))
            outputs['matrix2_dot'] = os.path.abspath(os.path.join(out_dir, 'fdt_matrix2.dot'))
        if isdefined(self.inputs.omatrix3):
            outputs['matrix3_dot'] = os.path.abspath(os.path.join(out_dir, 'fdt_matrix3.dot'))
        return outputs