from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.mark.parametrize('data, args, result', sampling_data_args_results)
def test_sampling(self, data, args, result):
    dos = RawDOSData(data[0], data[1])
    weights = dos._sample(*args[:-1], **args[-1])
    assert np.allclose(weights, result)
    with pytest.raises(ValueError):
        dos._sample([1], smearing="Gauss's spherical cousin")