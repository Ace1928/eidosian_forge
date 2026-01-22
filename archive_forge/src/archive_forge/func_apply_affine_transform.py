import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
@keras_export('keras._legacy.preprocessing.image.apply_affine_transform')
def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0, order=1):
    """Applies an affine transformation specified by the parameters given.

    DEPRECATED.
    """
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError("'row_axis', 'col_axis', and 'channel_axis' must be distinct")
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(f"Invalid axis' indices: {actual_indices - valid_indices}")
    if x.ndim != 3:
        raise ValueError('Input arrays must be multi-channel 2D images.')
    if channel_axis not in [0, 2]:
        raise ValueError('Channels are allowed and the first and last dimensions.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        transform_matrix = rotation_matrix
    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)
    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)
    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)
    if transform_matrix is not None:
        h, w = (x.shape[row_axis], x.shape[col_axis])
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [scipy.ndimage.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x