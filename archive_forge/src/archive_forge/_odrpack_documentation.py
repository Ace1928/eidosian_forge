import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
 Restarts the run with iter more iterations.

        Parameters
        ----------
        iter : int, optional
            ODRPACK's default for the number of new iterations is 10.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        