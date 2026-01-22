import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
def _get_valid_range(self):
    """Return valid range for image data

        The valid range can come from the image 'valid_range' or
        failing that, from the data type range
        """
    ddt = self.get_data_dtype()
    info = np.iinfo(ddt.type)
    try:
        valid_range = self._image.attrs['valid_range']
    except (AttributeError, KeyError):
        valid_range = [info.min, info.max]
    else:
        if valid_range[0] < info.min or valid_range[1] > info.max:
            raise ValueError('Valid range outside input data type range')
    return np.asarray(valid_range, dtype=np.float64)