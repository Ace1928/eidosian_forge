import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join
import numpy as np
from joblib import Memory
from ..utils import Bunch
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ._base import (
def _load_imgs(file_paths, slice_, color, resize):
    """Internally used to load images"""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('The Python Imaging Library (PIL) is required to load data from jpeg files. Please refer to https://pillow.readthedocs.io/en/stable/installation.html for installing PIL.')
    default_slice = (slice(0, 250), slice(0, 250))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple((s or ds for s, ds in zip(slice_, default_slice)))
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)
    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)
    n_faces = len(file_paths)
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)
    for i, file_path in enumerate(file_paths):
        if i % 1000 == 0:
            logger.debug('Loading face #%05d / %05d', i + 1, n_faces)
        pil_img = Image.open(file_path)
        pil_img = pil_img.crop((w_slice.start, h_slice.start, w_slice.stop, h_slice.stop))
        if resize is not None:
            pil_img = pil_img.resize((w, h))
        face = np.asarray(pil_img, dtype=np.float32)
        if face.ndim == 0:
            raise RuntimeError('Failed to read the image file %s, Please make sure that libjpeg is installed' % file_path)
        face /= 255.0
        if not color:
            face = face.mean(axis=2)
        faces[i, ...] = face
    return faces