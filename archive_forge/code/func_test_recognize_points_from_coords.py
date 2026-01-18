import pytest
import numpy as np
from ase.dft.kpoints import resolve_custom_points
def test_recognize_points_from_coords(special_points):
    path, dct = resolve_custom_points([[special_points['A'], special_points['B']]], special_points, 1e-05)
    assert path == 'AB'
    assert set(dct) == set('AB')