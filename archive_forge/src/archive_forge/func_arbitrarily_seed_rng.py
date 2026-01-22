import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE, check_output
import zlib
import pytest
import numpy as np
import ase
from ase.utils import workdir, seterr
from ase.test.factories import (CalculatorInputs,
from ase.dependencies import all_dependencies
@pytest.fixture(autouse=True)
def arbitrarily_seed_rng(request):
    ase_path = ase.__path__[0]
    abspath = Path(request.module.__file__)
    relpath = abspath.relative_to(ase_path)
    module_identifier = relpath.as_posix()
    function_name = request.function.__name__
    hashable_string = f'{module_identifier}:{function_name}'
    seed = zlib.adler32(hashable_string.encode('ascii')) % 12345
    state = np.random.get_state()
    np.random.seed(seed)
    yield
    print(f'Global seed for "{hashable_string}" was: {seed}')
    np.random.set_state(state)