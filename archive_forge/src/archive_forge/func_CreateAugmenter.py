import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
def CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0, saturation=0, hue=0, pca_noise=0, rand_gray=0, inter_method=2):
    """Creates an augmenter list.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    resize : int
        Resize shorter edge if larger than 0 at the begining
    rand_crop : bool
        Whether to enable random cropping other than center crop
    rand_resize : bool
        Whether to enable random sized cropping, require rand_crop to be enabled
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels, the number
        of channels will not be reduced to 1
    rand_mirror : bool
        Whether to apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        Mean pixel values for [r, g, b]
    std : np.ndarray or None
        Standard deviations for [r, g, b]
    brightness : float
        Brightness jittering range (percent)
    contrast : float
        Contrast jittering range (percent)
    saturation : float
        Saturation jittering range (percent)
    hue : float
        Hue jittering range (percent)
    pca_noise : float
        Pca noise level (percent)
    inter_method : int, default=2(Area-based)
        Interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.image.CreateAugmenter(data_shape=(3, 300, 300), rand_mirror=True,
    ...    mean=True, brightness=0.125, contrast=0.125, rand_gray=0.05,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10)
    >>> # dump the details
    >>> for aug in augs:
    ...    aug.dumps()
    """
    auglist = []
    if resize > 0:
        auglist.append(ResizeAug(resize, inter_method))
    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        auglist.append(RandomSizedCropAug(crop_size, 0.08, (3.0 / 4.0, 4.0 / 3.0), inter_method))
    elif rand_crop:
        auglist.append(RandomCropAug(crop_size, inter_method))
    else:
        auglist.append(CenterCropAug(crop_size, inter_method))
    if rand_mirror:
        auglist.append(HorizontalFlipAug(0.5))
    auglist.append(CastAug())
    if brightness or contrast or saturation:
        auglist.append(ColorJitterAug(brightness, contrast, saturation))
    if hue:
        auglist.append(HueJitterAug(hue))
    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.814], [-0.5836, -0.6948, 0.4203]])
        auglist.append(LightingAug(pca_noise, eigval, eigvec))
    if rand_gray > 0:
        auglist.append(RandomGrayAug(rand_gray))
    if mean is True:
        mean = nd.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, (np.ndarray, nd.NDArray)) and mean.shape[0] in [1, 3]
    if std is True:
        std = nd.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, (np.ndarray, nd.NDArray)) and std.shape[0] in [1, 3]
    if mean is not None or std is not None:
        auglist.append(ColorNormalizeAug(mean, std))
    return auglist