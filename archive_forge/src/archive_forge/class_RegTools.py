import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegTools(NiftyRegCommand):
    """Interface for executable reg_tools from NiftyReg platform.

    Tool delivering various actions related to registration such as
    resampling the input image to a chosen resolution or remove the nan and
    inf in the input image by a specified value.

    `Source code <https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyreg
    >>> node = niftyreg.RegTools()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.mul_val = 4
    >>> node.inputs.omp_core_val = 4
    >>> node.cmdline
    'reg_tools -in im1.nii -mul 4.0 -omp 4 -out im1_tools.nii.gz'

    """
    _cmd = get_custom_path('reg_tools')
    input_spec = RegToolsInputSpec
    output_spec = RegToolsOutputSpec
    _suffix = '_tools'

    def _format_arg(self, name, spec, value):
        if name == 'inter_val':
            inter_val = {'NN': 0, 'LIN': 1, 'CUB': 3, 'SINC': 4}
            return spec.argstr % inter_val[value]
        else:
            return super(RegTools, self)._format_arg(name, spec, value)