from nipype.interfaces.base import (
import os
class ResampleScalarVolume(SEMLikeCommandLine):
    """title: Resample Scalar Volume

    category: Legacy.Filtering

    description: Resampling an image is an important task in image analysis. It is especially important in the frame of image registration. This module implements image resampling through the use of itk Transforms. This module uses an Identity Transform. The resampling is controlled by the Output Spacing. "Resampling" is performed in space coordinates, not pixel/grid coordinates. It is quite important to ensure that image spacing is properly set on the images involved. The interpolator is required since the mapping from one space to the other will often require evaluation of the intensity of the image at non-grid positions. Several interpolators are available: linear, nearest neighbor, bspline and five flavors of sinc. The sinc interpolators, although more precise, are much slower than the linear and nearest neighbor interpolator. To resample label volumnes, nearest neighbor interpolation should be used exclusively.

    version: 0.1.0.$Revision: 20594 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ResampleVolume

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = ResampleScalarVolumeInputSpec
    output_spec = ResampleScalarVolumeOutputSpec
    _cmd = 'ResampleScalarVolume '
    _outputs_filenames = {'OutputVolume': 'OutputVolume.nii'}