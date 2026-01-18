import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('name', list(all_outputs))
def test_print_props(name):
    print(all_outputs[name])