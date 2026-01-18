import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extr_global_no_shared(self):
    wanted_global = [[-4.0 / 6], [-1.0 / 6], [3.0 / 6], [-1.0 / 6], [-7.0 / 6]]
    for i, vals in enumerate(self.base_wanted):
        wanted_global[i].extend(vals)
    iface.SignalExtraction(in_file=self.filenames['in_file'], label_files=self.filenames['label_files'], class_labels=self.labels, include_global=True, incl_shared_variance=False).run()
    self.assert_expected_output(self.global_labels, wanted_global)