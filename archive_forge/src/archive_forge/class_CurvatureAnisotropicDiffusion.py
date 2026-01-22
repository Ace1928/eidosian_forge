from nipype.interfaces.base import (
import os
class CurvatureAnisotropicDiffusion(SEMLikeCommandLine):
    """title: Curvature Anisotropic Diffusion

    category: Filtering.Denoising

    description: Performs anisotropic diffusion on an image using a modified curvature diffusion equation (MCDE).

    MCDE does not exhibit the edge enhancing properties of classic anisotropic diffusion, which can under certain conditions undergo a 'negative' diffusion, which enhances the contrast of edges.  Equations of the form of MCDE always undergo positive diffusion, with the conductance term only varying the strength of that diffusion.

     Qualitatively, MCDE compares well with other non-linear diffusion techniques.  It is less sensitive to contrast than classic Perona-Malik style diffusion, and preserves finer detailed structures in images.  There is a potential speed trade-off for using this function in place of Gradient Anisotropic Diffusion.  Each iteration of the solution takes roughly twice as long.  Fewer iterations, however, may be required to reach an acceptable solution.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/CurvatureAnisotropicDiffusion

    contributor: Bill Lorensen (GE)

    acknowledgements: This command module was derived from Insight/Examples (copyright) Insight Software Consortium
    """
    input_spec = CurvatureAnisotropicDiffusionInputSpec
    output_spec = CurvatureAnisotropicDiffusionOutputSpec
    _cmd = 'CurvatureAnisotropicDiffusion '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}