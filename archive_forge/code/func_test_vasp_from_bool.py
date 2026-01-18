import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def test_vasp_from_bool():
    for s in ('T', '.true.'):
        assert _from_vasp_bool(s) is True
    for s in ('f', '.False.'):
        assert _from_vasp_bool(s) is False
    with pytest.raises(ValueError):
        _from_vasp_bool('yes')
    with pytest.raises(AssertionError):
        _from_vasp_bool(True)