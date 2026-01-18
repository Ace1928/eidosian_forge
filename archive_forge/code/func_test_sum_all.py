import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.mark.parametrize('collection_data, collection_info, expected', zip(collection_data, collection_info, expected_sum))
def test_sum_all(self, collection_data, collection_info, expected):
    dc = DOSCollection([RawDOSData(*item, info=info) for item, info in zip(collection_data, collection_info)])
    summed_dc = dc.sum_all()
    energies, weights, ref_info = expected
    assert np.allclose(summed_dc.get_energies(), energies)
    assert np.allclose(summed_dc.get_weights(), weights)
    assert summed_dc.info == ref_info