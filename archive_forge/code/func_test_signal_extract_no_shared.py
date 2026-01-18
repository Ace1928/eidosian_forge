import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extract_no_shared(self):
    iface.SignalExtraction(in_file=self.filenames['in_file'], label_files=self.filenames['label_files'], class_labels=self.labels, incl_shared_variance=False).run()
    self.assert_expected_output(self.labels, self.base_wanted)