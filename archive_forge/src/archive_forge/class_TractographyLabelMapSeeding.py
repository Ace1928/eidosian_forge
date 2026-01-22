from nipype.interfaces.base import (
import os
class TractographyLabelMapSeeding(SEMLikeCommandLine):
    """title: Tractography Label Map Seeding

    category: Diffusion.Diffusion Tensor Images

    description: Seed tracts on a Diffusion Tensor Image (DT) from a label map

    version: 0.1.0.$Revision: 1892 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Seeding

    license: slicer3

    contributor: Raul San Jose (SPL, BWH), Demian Wassermann (SPL, BWH)

    acknowledgements: Laboratory of Mathematics in Imaging. This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = TractographyLabelMapSeedingInputSpec
    output_spec = TractographyLabelMapSeedingOutputSpec
    _cmd = 'TractographyLabelMapSeeding '
    _outputs_filenames = {'OutputFibers': 'OutputFibers.vtk', 'outputdirectory': 'outputdirectory'}