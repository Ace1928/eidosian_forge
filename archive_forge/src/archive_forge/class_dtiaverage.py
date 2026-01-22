import os
from ...base import (
class dtiaverage(SEMLikeCommandLine):
    """title: DTIAverage (DTIProcess)

    category: Diffusion.Diffusion Tensor Images.CommandLineOnly

    description: dtiaverage is a program that allows to compute the average of an arbitrary number of tensor fields (listed after the --inputs option) This program is used in our pipeline as the last step of the atlas building processing. When all the tensor fields have been deformed in the same space, to create the average tensor field (--tensor_output) we use dtiaverage.
     Several average method can be used (specified by the --method option): euclidean, log-euclidean and pga. The default being euclidean.

    version: 1.0.0

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/DTIProcess

    license: Copyright (c)  Casey Goodlett. All rights reserved.
        See http://www.ia.unc.edu/dev/Copyright.htm for details.
        This software is distributed WITHOUT ANY WARRANTY; without even
        the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
        PURPOSE.  See the above copyright notices for more information.

    contributor: Casey Goodlett
    """
    input_spec = dtiaverageInputSpec
    output_spec = dtiaverageOutputSpec
    _cmd = ' dtiaverage '
    _outputs_filenames = {'tensor_output': 'tensor_output.nii'}
    _redirect_x = False