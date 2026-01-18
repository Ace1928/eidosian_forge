import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def write_geometry(filepath, coords, faces, create_stamp=None, volume_info=None):
    """Write a triangular format Freesurfer surface mesh.

    Parameters
    ----------
    filepath : str
        Path to surface file.
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    create_stamp : str, optional
        User/time stamp (default: "created by <user> on <ctime>")
    volume_info : dict-like or None, optional
        Key-value pairs to encode at the end of the file.

        Valid keys:

        * 'head' : array of int
        * 'valid' : str
        * 'filename' : str
        * 'volume' : array of int, shape (3,)
        * 'voxelsize' : array of float, shape (3,)
        * 'xras' : array of float, shape (3,)
        * 'yras' : array of float, shape (3,)
        * 'zras' : array of float, shape (3,)
        * 'cras' : array of float, shape (3,)

    """
    magic_bytes = np.array([255, 255, 254], dtype=np.uint8)
    if create_stamp is None:
        create_stamp = f'created by {getpass.getuser()} on {time.ctime()}'
    with open(filepath, 'wb') as fobj:
        magic_bytes.tofile(fobj)
        fobj.write(f'{create_stamp}\n\n'.encode())
        np.array([coords.shape[0], faces.shape[0]], dtype='>i4').tofile(fobj)
        coords.astype('>f4').reshape(-1).tofile(fobj)
        faces.astype('>i4').reshape(-1).tofile(fobj)
        if volume_info is not None and len(volume_info) > 0:
            fobj.write(_serialize_volume_info(volume_info))