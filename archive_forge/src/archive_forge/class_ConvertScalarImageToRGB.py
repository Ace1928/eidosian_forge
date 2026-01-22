import os
from ..base import TraitedSpec, File, traits
from .base import ANTSCommand, ANTSCommandInputSpec
class ConvertScalarImageToRGB(ANTSCommand):
    """
    Convert scalar images to RGB.

    Examples
    --------
    >>> from nipype.interfaces.ants.visualization import ConvertScalarImageToRGB
    >>> converter = ConvertScalarImageToRGB()
    >>> converter.inputs.dimension = 3
    >>> converter.inputs.input_image = 'T1.nii.gz'
    >>> converter.inputs.colormap = 'jet'
    >>> converter.inputs.minimum_input = 0
    >>> converter.inputs.maximum_input = 6
    >>> converter.cmdline
    'ConvertScalarImageToRGB 3 T1.nii.gz rgb.nii.gz none jet none 0 6 0 255'

    """
    _cmd = 'ConvertScalarImageToRGB'
    input_spec = ConvertScalarImageToRGBInputSpec
    output_spec = ConvertScalarImageToRGBOutputSpec

    def _format_arg(self, opt, spec, val):
        return super(ConvertScalarImageToRGB, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = os.path.join(os.getcwd(), self.inputs.output_image)
        return outputs