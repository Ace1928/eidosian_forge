import os.path as op
import nibabel as nb
import numpy as np
from nibabel.volumeutils import native_code
from nibabel.orientations import aff2axcodes
from ... import logging
from ...utils.filemanip import split_filename
from ..base import TraitedSpec, File, isdefined
from ..dipy.base import DipyBaseInterface, HAVE_DIPY as have_dipy
def transform_to_affine(streams, header, affine):
    try:
        from dipy.tracking.utils import transform_tracking_output
    except ImportError:
        from dipy.tracking.utils import move_streamlines as transform_tracking_output
    rotation, scale = np.linalg.qr(affine)
    streams = transform_tracking_output(streams, rotation)
    scale[0:3, 0:3] = np.dot(scale[0:3, 0:3], np.diag(1.0 / header['voxel_size']))
    scale[0:3, 3] = abs(scale[0:3, 3])
    streams = transform_tracking_output(streams, scale)
    return streams