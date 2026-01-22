from nipype.interfaces.base import (
import os
class DWIToDTIEstimation(SEMLikeCommandLine):
    """title: DWI to DTI Estimation

    category: Diffusion.Diffusion Weighted Images

    description: Performs a tensor model estimation from diffusion weighted images.

    There are three estimation methods available: least squares, weighted least squares and non-linear estimation. The first method is the traditional method for tensor estimation and the fastest one. Weighted least squares takes into account the noise characteristics of the MRI images to weight the DWI samples used in the estimation based on its intensity magnitude. The last method is the more complex.

    version: 0.1.0.$Revision: 1892 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DiffusionTensorEstimation

    license: slicer3

    contributor: Raul San Jose (SPL, BWH)

    acknowledgements: This command module is based on the estimation functionality provided by the Teem library. This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = DWIToDTIEstimationInputSpec
    output_spec = DWIToDTIEstimationOutputSpec
    _cmd = 'DWIToDTIEstimation '
    _outputs_filenames = {'outputTensor': 'outputTensor.nii', 'outputBaseline': 'outputBaseline.nii'}