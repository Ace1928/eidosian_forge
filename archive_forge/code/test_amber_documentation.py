import subprocess
from ase import Atoms
from ase.calculators.amber import Amber
Test that amber calculator works.

    This is conditional on the existence of the $AMBERHOME/bin/sander
    executable.
    