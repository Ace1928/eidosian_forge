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
def _strict_sort_order(self):
    """Determine the sort order based on several image definition fields.

        The fields taken into consideration, if present, are (in order from
        slowest to fastest variation after sorting):

            - image_defs['image_type_mr']                # Re, Im, Mag, Phase
            - image_defs['dynamic scan number']          # repetition
            - image_defs['label type']                   # ASL tag/control
            - image_defs['diffusion b value number']     # diffusion b value
            - image_defs['gradient orientation number']  # diffusion directoin
            - image_defs['cardiac phase number']         # cardiac phase
            - image_defs['echo number']                  # echo
            - image_defs['slice number']                 # slice

        Data sorting is done in two stages:

            1. an initial sort using the keys described above
            2. a resort after generating two additional sort keys:

                * a key to assign unique volume numbers to any volumes that
                  didn't have a unique sort based on the keys above
                  (see :func:`vol_numbers`).
                * a sort key based on `vol_is_full` to identify truncated
                  volumes

        A case where the initial sort may not create a unique label for each
        volume is diffusion scans acquired in the older V4 .PAR format, where
        diffusion direction info is not available.
        """
    idefs = self.image_defs
    slice_nos = idefs['slice number']
    dynamics = idefs['dynamic scan number']
    phases = idefs['cardiac phase number']
    echos = idefs['echo number']
    image_type = idefs['image_type_mr']
    asl_keys = (idefs['label type'],) if 'label type' in idefs.dtype.names else ()
    if self.general_info['diffusion'] != 0:
        bvals = self.get_def('diffusion b value number')
        if bvals is None:
            bvals = self.get_def('diffusion_b_factor')
        bvecs = self.get_def('gradient orientation number')
        if bvecs is None:
            diffusion_keys = (bvals,)
        else:
            diffusion_keys = (bvecs, bvals)
    else:
        diffusion_keys = ()
    keys = (slice_nos, echos, phases) + diffusion_keys + asl_keys + (dynamics, image_type)
    initial_sort_order = np.lexsort(keys)
    vol_nos = vol_numbers(slice_nos[initial_sort_order])
    is_full = vol_is_full(slice_nos[initial_sort_order], self.general_info['max_slices'])
    return initial_sort_order[np.lexsort((vol_nos, is_full))]