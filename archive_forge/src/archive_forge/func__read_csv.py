import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
def _read_csv(self):
    """
        Read from csv in_file and return an array and ROI names

        The input file should have a first row containing the names of the
        ROIs (strings)

        the rest of the data will be read in and transposed so that the rows
        (TRs) will becomes the second (and last) dimension of the array

        """
    first_row = open(self.inputs.in_file).readline()
    if not first_row[1].isalpha():
        raise ValueError('First row of in_file should contain ROI names as strings of characters')
    roi_names = open(self.inputs.in_file).readline().replace('"', '').strip('\n').split(',')
    data = np.loadtxt(self.inputs.in_file, skiprows=1, delimiter=',').T
    return (data, roi_names)