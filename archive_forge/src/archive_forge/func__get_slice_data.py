import os
import sys
import logging
import subprocess
from ..core import Format, BaseProgressIndicator, StdoutProgressIndicator
from ..core import read_n_bytes
def _get_slice_data(self, index):
    nslices = self._data.shape[0] if self._data.ndim == 3 else 1
    if nslices > 1:
        return (self._data[index], self._info)
    elif index == 0:
        return (self._data, self._info)
    else:
        raise IndexError('Dicom file contains only one slice.')