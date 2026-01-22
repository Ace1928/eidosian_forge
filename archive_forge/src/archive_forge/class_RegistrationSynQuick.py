import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class RegistrationSynQuick(ANTSCommand):
    """
    Registration using a symmetric image normalization method (SyN).
    You can read more in Avants et al.; Med Image Anal., 2008
    (https://www.ncbi.nlm.nih.gov/pubmed/17659998).

    Examples
    --------

    >>> from nipype.interfaces.ants import RegistrationSynQuick
    >>> reg = RegistrationSynQuick()
    >>> reg.inputs.fixed_image = 'fixed1.nii'
    >>> reg.inputs.moving_image = 'moving1.nii'
    >>> reg.inputs.num_threads = 2
    >>> reg.cmdline
    'antsRegistrationSyNQuick.sh -d 3 -f fixed1.nii -r 32 -m moving1.nii -n 2 -o transform -p d -s 26 -t s'
    >>> reg.run()  # doctest: +SKIP

    example for multiple images

    >>> from nipype.interfaces.ants import RegistrationSynQuick
    >>> reg = RegistrationSynQuick()
    >>> reg.inputs.fixed_image = ['fixed1.nii', 'fixed2.nii']
    >>> reg.inputs.moving_image = ['moving1.nii', 'moving2.nii']
    >>> reg.inputs.num_threads = 2
    >>> reg.cmdline
    'antsRegistrationSyNQuick.sh -d 3 -f fixed1.nii -f fixed2.nii -r 32 -m moving1.nii -m moving2.nii -n 2 -o transform -p d -s 26 -t s'
    >>> reg.run()  # doctest: +SKIP
    """
    _cmd = 'antsRegistrationSyNQuick.sh'
    input_spec = RegistrationSynQuickInputSpec
    output_spec = RegistrationSynQuickOutputSpec

    def _num_threads_update(self):
        """
        antsRegistrationSyNQuick.sh ignores environment variables,
        so override environment update from ANTSCommand class
        """
        pass

    def _format_arg(self, name, spec, value):
        if name == 'precision_type':
            return spec.argstr % value[0]
        return super(RegistrationSynQuick, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_base = os.path.abspath(self.inputs.output_prefix)
        outputs['warped_image'] = out_base + 'Warped.nii.gz'
        outputs['inverse_warped_image'] = out_base + 'InverseWarped.nii.gz'
        outputs['out_matrix'] = out_base + '0GenericAffine.mat'
        if self.inputs.transform_type not in ('t', 'r', 'a'):
            outputs['forward_warp_field'] = out_base + '1Warp.nii.gz'
            outputs['inverse_warp_field'] = out_base + '1InverseWarp.nii.gz'
        return outputs