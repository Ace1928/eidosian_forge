from nipype.interfaces.base import (
import os
class DWIJointRicianLMMSEFilter(SEMLikeCommandLine):
    """title: DWI Joint Rician LMMSE Filter

    category: Diffusion.Diffusion Weighted Images

    description: This module reduces Rician noise (or unwanted detail) on a set of diffusion weighted images. For this, it filters the image in the mean squared error sense using a Rician noise model. The N closest gradient directions to the direction being processed are filtered together to improve the results: the noise-free signal is seen as an n-diemensional vector which has to be estimated with the LMMSE method from a set of corrupted measurements. To that end, the covariance matrix of the noise-free vector and the cross covariance between this signal and the noise have to be estimated, which is done taking into account the image formation process.
    The noise parameter is automatically estimated from a rough segmentation of the background of the image. In this area the signal is simply 0, so that Rician statistics reduce to Rayleigh and the noise power can be easily estimated from the mode of the histogram.
    A complete description of the algorithm may be found in:
    Antonio Tristan-Vega and Santiago Aja-Fernandez, DWI filtering using joint information for DTI and HARDI, Medical Image Analysis, Volume 14, Issue 2, Pages 205-218. 2010.

    version: 0.1.1.$Revision: 1 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/JointRicianLMMSEImageFilter

    contributor: Antonio Tristan Vega (UVa), Santiago Aja Fernandez (UVa)

    acknowledgements: Partially founded by grant number TEC2007-67073/TCM from the Comision Interministerial de Ciencia y Tecnologia (Spain).
    """
    input_spec = DWIJointRicianLMMSEFilterInputSpec
    output_spec = DWIJointRicianLMMSEFilterOutputSpec
    _cmd = 'DWIJointRicianLMMSEFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}