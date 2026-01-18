import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data

    Checks whether writing the Cifti2 array to disc and reading it back in gives the same object

    Parameters
    ----------
    arr : array
        N-dimensional array of data
    axes : Sequence[cifti2_axes.Axis]
        sequence of length N with the meaning of the rows/columns along each dimension
    extension : str
        custom extension to use
    