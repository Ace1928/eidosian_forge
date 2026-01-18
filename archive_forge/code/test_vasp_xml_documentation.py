import numpy as np
import pytest
from ase.io import read

Run some VASP tests to ensure that the VASP calculator works. This
is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
environment variables

