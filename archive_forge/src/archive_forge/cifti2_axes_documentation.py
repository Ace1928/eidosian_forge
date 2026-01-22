from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2

        Gives the time point of a specific row/column

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        float
        