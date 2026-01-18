import os
import sys
import tempfile
from glob import glob
import numpy as np
import pytest
from ... import from_cmdstanpy
from ..helpers import (  # pylint: disable=unused-import
def test_sampler_stats(self, data, eight_schools_params):
    inference_data = self.get_inference_data(data, eight_schools_params)
    test_dict = {'sample_stats': ['lp', 'diverging']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails
    assert len(inference_data.sample_stats.lp.shape) == 2