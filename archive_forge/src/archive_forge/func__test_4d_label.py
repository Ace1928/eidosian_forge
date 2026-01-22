import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def _test_4d_label(self, wanted, fake_labels, include_global=False, incl_shared_variance=True):
    utils.save_toy_nii(fake_labels, self.filenames['4d_label_file'])
    iface.SignalExtraction(in_file=self.filenames['in_file'], label_files=self.filenames['4d_label_file'], class_labels=self.labels, incl_shared_variance=incl_shared_variance, include_global=include_global).run()
    wanted_labels = self.global_labels if include_global else self.labels
    self.assert_expected_output(wanted_labels, wanted)