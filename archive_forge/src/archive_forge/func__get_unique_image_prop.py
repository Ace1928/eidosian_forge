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
def _get_unique_image_prop(self, name):
    """Scan image definitions and return unique value of a property.

        * Get array for named field of ``self.image_defs``;
        * Check that all rows in the array are the same and raise error
          otherwise;
        * Return the row.

        Parameters
        ----------
        name : str
            Name of the property in ``self.image_defs``

        Returns
        -------
        unique_value : scalar or array

        Raises
        ------
        PARRECError
            if the rows of ``self.image_defs[name]`` do not all compare equal.
        """
    props = self.image_defs[name]
    if np.any(np.diff(props, axis=0)):
        raise PARRECError(f'Varying {name} in image sequence ({props}). This is not supported.')
    return props[0]