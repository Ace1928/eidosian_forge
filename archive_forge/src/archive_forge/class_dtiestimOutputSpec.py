import os
from ...base import (
class dtiestimOutputSpec(TraitedSpec):
    tensor_output = File(desc='Tensor OutputImage', exists=True)
    B0 = File(desc='Baseline image, average of all baseline images', exists=True)
    idwi = File(desc='idwi output image. Image with isotropic diffusion-weighted information = geometric mean of diffusion images', exists=True)
    B0_mask_output = File(desc='B0 mask used for the estimation. B0 thresholded either with the -t option value or the automatic OTSU value', exists=True)