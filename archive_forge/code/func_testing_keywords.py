import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
@pytest.fixture
def testing_keywords():
    kw_data = {}
    for kwt in kw_types:
        kwtlow = kwt.lower().replace(' ', '_')
        if 'Boolean' in kwt:
            kwtlow = 'boolean'
        kw = 'test_{0}_kw'.format(kwtlow)
        kw_data[kw] = {'docstring': 'A fake {0} keyword'.format(kwt), 'option_type': kwt, 'keyword': kw, 'level': 'Dummy'}
    param_kws = [('continuation', 'String'), ('reuse', 'String')]
    param_kw_data = {}
    for pkw, t in param_kws:
        param_kw_data[pkw] = {'docstring': 'Dummy {0} keyword'.format(pkw), 'option_type': t, 'keyword': pkw, 'level': 'Dummy'}
    param_kw_data.update(kw_data)
    cell_kws = [('species_pot', 'Block'), ('symmetry_ops', 'Block'), ('positions_abs_intermediate', 'Block'), ('positions_abs_product', 'Block'), ('positions_frac_intermediate', 'Block'), ('positions_frac_product', 'Block'), ('kpoint_mp_grid', 'Integer Vector'), ('kpoint_mp_offset', 'Real Vector'), ('kpoint_list', 'Block'), ('bs_kpoint_list', 'Block')]
    cell_kw_data = {}
    for ckw, t in cell_kws:
        cell_kw_data[ckw] = {'docstring': 'Dummy {0} keyword'.format(ckw), 'option_type': t, 'keyword': ckw, 'level': 'Dummy'}
    cell_kw_data.update(kw_data)
    param_dict = make_param_dict(param_kw_data)
    cell_dict = make_cell_dict(cell_kw_data)
    return CastepKeywords(param_dict, cell_dict, kw_types, kw_levels, 'Castep v.Fake')