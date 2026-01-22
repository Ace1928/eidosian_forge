import numpy as np
from ..util import img_as_float
from .._shared.utils import (
def _prepare_grayscale_input_nD(image):
    image = np.squeeze(image)
    check_nD(image, range(2, 6))
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)