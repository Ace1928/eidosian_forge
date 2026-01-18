import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extr_4d_global_no_shared(self):
    wanted_global = [[3.0 / 8], [-3.0 / 8], [1.0 / 8], [-7.0 / 8], [-9.0 / 8]]
    for i, vals in enumerate(self.fourd_wanted):
        wanted_global[i].extend(vals)
    self._test_4d_label(wanted_global, self.fake_4d_label_data, include_global=True, incl_shared_variance=False)