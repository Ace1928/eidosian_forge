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
def _process_gen_dict(gen_dict):
    """Process `gen_dict` key, values into `general_info`"""
    general_info = {}
    for key, value in gen_dict.items():
        props = _hdr_key_dict[key]
        if len(props) == 2:
            value = props[1](value)
        elif len(props) == 3:
            value = np.fromstring(value, props[1], sep=' ')
            if props[2] is not None:
                value.shape = props[2]
        general_info[props[0]] = value
    return general_info