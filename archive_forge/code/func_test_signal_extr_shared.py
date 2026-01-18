import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extr_shared(self):
    wanted = []
    for vol in range(self.fake_fmri_data.shape[3]):
        volume = self.fake_fmri_data[:, :, :, vol].flatten()
        wanted_row = []
        for reg in range(self.fake_4d_label_data.shape[3]):
            region = self.fake_4d_label_data[:, :, :, reg].flatten()
            wanted_row.append((volume * region).sum() / (region * region).sum())
        wanted.append(wanted_row)
    self._test_4d_label(wanted, self.fake_4d_label_data)