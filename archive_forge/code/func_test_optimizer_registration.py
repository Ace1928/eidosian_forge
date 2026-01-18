import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_optimizer_registration():

    def custom_optimizer(inputs, output, size_dict, memory_limit):
        return [(0, 1)] * (len(inputs) - 1)
    with pytest.raises(KeyError):
        oe.paths.register_path_fn('optimal', custom_optimizer)
    oe.paths.register_path_fn('custom', custom_optimizer)
    assert 'custom' in oe.paths._PATH_OPTIONS
    eq = 'ab,bc,cd'
    shapes = [(2, 3), (3, 4), (4, 5)]
    path, path_info = oe.contract_path(eq, *shapes, shapes=True, optimize='custom')
    assert path == [(0, 1), (0, 1)]
    del oe.paths._PATH_OPTIONS['custom']