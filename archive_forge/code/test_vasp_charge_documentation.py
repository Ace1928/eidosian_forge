import pytest
from ase.build import bulk

    Run VASP tests to ensure that determining number of electrons from
    user-supplied charge works correctly.

    Test that the number of charge found matches the expected.
    