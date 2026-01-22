from nipype.interfaces.base import (
import os
class ResampleScalarVectorDWIVolume(SEMLikeCommandLine):
    """title: Resample Scalar/Vector/DWI Volume

    category: Filtering

    description: This module implements image and vector-image resampling through  the use of itk Transforms.It can also handle diffusion weighted MRI image resampling. "Resampling" is performed in space coordinates, not pixel/grid coordinates. It is quite important to ensure that image spacing is properly set on the images involved. The interpolator is required since the mapping from one space to the other will often require evaluation of the intensity of the image at non-grid positions.

    Warning: To resample DWMR Images, use nrrd input and output files.

    Warning: Do not use to resample Diffusion Tensor Images, tensors would  not be reoriented

    version: 0.1

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ResampleScalarVectorDWIVolume

    contributor: Francois Budin (UNC)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149. Information on the National Centers for Biomedical Computing can be obtained from http://nihroadmap.nih.gov/bioinformatics
    """
    input_spec = ResampleScalarVectorDWIVolumeInputSpec
    output_spec = ResampleScalarVectorDWIVolumeOutputSpec
    _cmd = 'ResampleScalarVectorDWIVolume '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}