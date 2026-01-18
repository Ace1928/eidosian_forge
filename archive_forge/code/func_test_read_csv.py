import os
import tempfile
import numpy as np
import pytest
from nipype.testing import example_data
import nipype.interfaces.nitime as nitime
@pytest.mark.skipif(no_nitime, reason='nitime is not installed')
def test_read_csv():
    """Test that reading the data from csv file gives you back a reasonable
    time-series object"""
    CA = nitime.CoherenceAnalyzer()
    CA.inputs.TR = 1.89
    CA.inputs.in_file = example_data('fmri_timeseries_nolabels.csv')
    with pytest.raises(ValueError):
        CA._read_csv()
    CA.inputs.in_file = example_data('fmri_timeseries.csv')
    data, roi_names = CA._read_csv()
    assert data[0][0] == 10125.9
    assert roi_names[0] == 'WM'