import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
@pytest.mark.parametrize('args, expected_len', [(['a', 'b', '#', 'c'], 2), (['a', 'b', '!', 'c', '#', 'd'], 2), (['#', 'a', 'b', '!', 'c', '#', 'd'], 0)])
def test_vasp_args_without_comment(args, expected_len):
    """Test comment splitting logic"""
    clean_args = _args_without_comment(args)
    assert len(clean_args) == expected_len