import pytest
import numpy as np
from ase.lattice import (get_lattice_from_canonical_cell, all_variants,
Bravais lattice type check.

1) For each Bravais variant, check that we recognize the
standard cell correctly.

2) For those Bravais lattices that we can recognize in non-standard form,
   Niggli-reduce them and recognize them as well.