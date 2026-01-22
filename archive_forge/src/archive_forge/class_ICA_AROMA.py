from ..base import (
import os
class ICA_AROMA(CommandLine):
    """
    Interface for the ICA_AROMA.py script.

    ICA-AROMA (i.e. 'ICA-based Automatic Removal Of Motion Artifacts') concerns
    a data-driven method to identify and remove motion-related independent
    components from fMRI data. To that end it exploits a small, but robust
    set of theoretically motivated features, preventing the need for classifier
    re-training and therefore providing direct and easy applicability.

    See link for further documentation: https://github.com/rhr-pruim/ICA-AROMA

    Example
    -------

    >>> from nipype.interfaces.fsl import ICA_AROMA
    >>> from nipype.testing import example_data
    >>> AROMA_obj = ICA_AROMA()
    >>> AROMA_obj.inputs.in_file = 'functional.nii'
    >>> AROMA_obj.inputs.mat_file = 'func_to_struct.mat'
    >>> AROMA_obj.inputs.fnirt_warp_file = 'warpfield.nii'
    >>> AROMA_obj.inputs.motion_parameters = 'fsl_mcflirt_movpar.txt'
    >>> AROMA_obj.inputs.mask = 'mask.nii.gz'
    >>> AROMA_obj.inputs.denoise_type = 'both'
    >>> AROMA_obj.inputs.out_dir = 'ICA_testout'
    >>> AROMA_obj.cmdline  # doctest: +ELLIPSIS
    'ICA_AROMA.py -den both -warp warpfield.nii -i functional.nii -m mask.nii.gz -affmat func_to_struct.mat -mc fsl_mcflirt_movpar.txt -o .../ICA_testout'
    """
    _cmd = 'ICA_AROMA.py'
    input_spec = ICA_AROMAInputSpec
    output_spec = ICA_AROMAOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'out_dir':
            return trait_spec.argstr % os.path.abspath(value)
        return super(ICA_AROMA, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dir'] = os.path.abspath(self.inputs.out_dir)
        out_dir = outputs['out_dir']
        if self.inputs.denoise_type in ('aggr', 'both'):
            outputs['aggr_denoised_file'] = os.path.join(out_dir, 'denoised_func_data_aggr.nii.gz')
        if self.inputs.denoise_type in ('nonaggr', 'both'):
            outputs['nonaggr_denoised_file'] = os.path.join(out_dir, 'denoised_func_data_nonaggr.nii.gz')
        return outputs