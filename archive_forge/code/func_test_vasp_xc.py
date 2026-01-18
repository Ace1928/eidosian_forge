import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def test_vasp_xc(vaspinput_factory):
    """
    Run some tests to ensure that the xc setting in the VASP calculator
    works.
    """
    calc_vdw = vaspinput_factory(xc='optb86b-vdw')
    assert dict_is_subset({'param1': 0.1234, 'param2': 1.0}, calc_vdw.float_params)
    assert calc_vdw.bool_params['luse_vdw'] is True
    calc_hse = vaspinput_factory(xc='hse06', hfscreen=0.1, gga='RE', encut=400, sigma=0.5)
    assert dict_is_subset({'hfscreen': 0.1, 'encut': 400, 'sigma': 0.5}, calc_hse.float_params)
    assert calc_hse.bool_params['lhfcalc'] is True
    assert dict_is_subset({'gga': 'RE'}, calc_hse.string_params)
    calc_pw91 = vaspinput_factory(xc='pw91', kpts=(2, 2, 2), gamma=True, lreal='Auto')
    assert dict_is_subset({'pp': 'PW91', 'kpts': (2, 2, 2), 'gamma': True, 'reciprocal': False}, calc_pw91.input_params)