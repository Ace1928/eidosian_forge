import pytest
import numpy as np
from ase import io
from ase.optimize import BFGS
from ase.build import bulk

    Run VASP tests to ensure that relaxation with the VASP calculator works.
    This is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
    environment variables.

    