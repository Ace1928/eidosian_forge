import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_sample_stats(self, paths):
    for key, path in paths.items():
        if 'missing' in key:
            continue
        inference_data = self.get_inference_data(path)
        assert hasattr(inference_data, 'sample_stats')
        assert 'step_size' in inference_data.sample_stats.attrs
        assert inference_data.sample_stats.attrs['step_size'] == 'stepsize'