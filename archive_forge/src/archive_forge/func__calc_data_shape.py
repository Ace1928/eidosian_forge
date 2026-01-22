import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def _calc_data_shape(self):
    """Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        n_inplaneX : int
            number of voxels in X direction.
        n_inplaneY : int
            number of voxels in Y direction.
        n_slices : int
            number of slices.
        n_vols : int
            number of volumes or absent for 3D image.

        Notes
        -----
        This routine gets called in ``__init__``, so may not be able to use
        some attributes available in the fully initialized object.
        """
    inplane_shape = tuple(self._get_unique_image_prop('recon resolution'))
    shape = inplane_shape + (self._get_n_slices(),)
    n_vols = self._get_n_vols()
    return shape + (n_vols,) if n_vols > 1 else shape