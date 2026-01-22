import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
def _sin_flow_gen(image0, max_motion=4.5, npics=5):
    """Generate a synthetic ground truth optical flow with a sinusoid as
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

    """
    grid = np.meshgrid(*[np.arange(n) for n in image0.shape], indexing='ij')
    grid = np.stack(grid)
    gt_flow = np.zeros_like(grid, dtype=float)
    gt_flow[0, ...] = max_motion * np.sin(grid[0] / grid[0].max() * npics * np.pi)
    image1 = warp(image0, grid - gt_flow, mode='edge')
    return (gt_flow, image1)