from nipype.interfaces.base import (
import os
class GradientAnisotropicDiffusion(SEMLikeCommandLine):
    """title: Gradient Anisotropic Diffusion

    category: Filtering.Denoising

    description: Runs gradient anisotropic diffusion on a volume.

    Anisotropic diffusion methods reduce noise (or unwanted detail) in images while preserving specific image features, like edges.  For many applications, there is an assumption that light-dark transitions (edges) are interesting.  Standard isotropic diffusion methods move and blur light-dark boundaries.  Anisotropic diffusion methods are formulated to specifically preserve edges. The conductance term for this implementation is a function of the gradient magnitude of the image at each point, reducing the strength of diffusion at edges. The numerical implementation of this equation is similar to that described in the Perona-Malik paper, but uses a more robust technique for gradient magnitude estimation and has been generalized to N-dimensions.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/GradientAnisotropicDiffusion

    contributor: Bill Lorensen (GE)

    acknowledgements: This command module was derived from Insight/Examples (copyright) Insight Software Consortium
    """
    input_spec = GradientAnisotropicDiffusionInputSpec
    output_spec = GradientAnisotropicDiffusionOutputSpec
    _cmd = 'GradientAnisotropicDiffusion '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}