import os
from ..base import TraitedSpec, File, traits
from .base import ANTSCommand, ANTSCommandInputSpec
class CreateTiledMosaic(ANTSCommand):
    """The program CreateTiledMosaic in conjunction with ConvertScalarImageToRGB
    provides useful functionality for common image analysis tasks. The basic
    usage of CreateTiledMosaic is to tile a 3-D image volume slice-wise into
    a 2-D image.

    Examples
    --------

    >>> from nipype.interfaces.ants.visualization import CreateTiledMosaic
    >>> mosaic_slicer = CreateTiledMosaic()
    >>> mosaic_slicer.inputs.input_image = 'T1.nii.gz'
    >>> mosaic_slicer.inputs.rgb_image = 'rgb.nii.gz'
    >>> mosaic_slicer.inputs.mask_image = 'mask.nii.gz'
    >>> mosaic_slicer.inputs.output_image = 'output.png'
    >>> mosaic_slicer.inputs.alpha_value = 0.5
    >>> mosaic_slicer.inputs.direction = 2
    >>> mosaic_slicer.inputs.pad_or_crop = '[ -15x -50 , -15x -30 ,0]'
    >>> mosaic_slicer.inputs.slices = '[2 ,100 ,160]'
    >>> mosaic_slicer.cmdline
    'CreateTiledMosaic -a 0.50 -d 2 -i T1.nii.gz -x mask.nii.gz -o output.png -p [ -15x -50 , -15x -30 ,0] -r rgb.nii.gz -s [2 ,100 ,160]'
    """
    _cmd = 'CreateTiledMosaic'
    input_spec = CreateTiledMosaicInputSpec
    output_spec = CreateTiledMosaicOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = os.path.join(os.getcwd(), self.inputs.output_image)
        return outputs