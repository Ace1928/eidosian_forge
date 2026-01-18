import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        