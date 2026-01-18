import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
Generate a synthetic ground truth optical flow with a sinusoid as
      first component.

    Parameters
    ----------
    image0: ndarray
        The base image to be warped.
    max_motion: float
        Maximum flow magnitude.
    npics: int
        Number of sinusoid pics.

    Returns
    -------
    flow, image1 : ndarray
        The synthetic ground truth optical flow with a sinusoid as
        first component and the corresponding warped image.

    