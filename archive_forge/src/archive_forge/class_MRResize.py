import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRResize(MRTrix3Base):
    """
    Resize an image by defining the new image resolution, voxel size or a
    scale factor. If the image is 4D, then only the first 3 dimensions can be
    resized. Also, if the image is down-sampled, the appropriate smoothing is
    automatically applied using Gaussian smoothing.
    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/mrresize.html>

    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt

    Defining the new image resolution:
    >>> image_resize = mrt.MRResize()
    >>> image_resize.inputs.in_file = 'dwi.mif'
    >>> image_resize.inputs.image_size = (256, 256, 144)
    >>> image_resize.cmdline                               # doctest: +ELLIPSIS
    'mrresize -size 256,256,144 -interp cubic dwi.mif dwi_resized.mif'
    >>> image_resize.run()                                 # doctest: +SKIP

    Defining the new image's voxel size:
    >>> voxel_resize = mrt.MRResize()
    >>> voxel_resize.inputs.in_file = 'dwi.mif'
    >>> voxel_resize.inputs.voxel_size = (1, 1, 1)
    >>> voxel_resize.cmdline                               # doctest: +ELLIPSIS
    'mrresize -interp cubic -voxel 1,1,1 dwi.mif dwi_resized.mif'
    >>> voxel_resize.run()                                 # doctest: +SKIP

    Defining the scale factor of each image dimension:
    >>> scale_resize = mrt.MRResize()
    >>> scale_resize.inputs.in_file = 'dwi.mif'
    >>> scale_resize.inputs.scale_factor = (0.5,0.5,0.5)
    >>> scale_resize.cmdline                               # doctest: +ELLIPSIS
    'mrresize -interp cubic -scale 0.5,0.5,0.5 dwi.mif dwi_resized.mif'
    >>> scale_resize.run()                                 # doctest: +SKIP
    """
    _cmd = 'mrresize'
    input_spec = MRResizeInputSpec
    output_spec = MRResizeOutputSpec