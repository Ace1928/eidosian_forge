from __future__ import annotations
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import polar
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, fast_norm
from pymatgen.core.interface import Interface, label_termination
from pymatgen.core.surface import SlabGenerator
def _find_terminations(self):
    """Finds all terminations."""
    film_sg = SlabGenerator(self.film_structure, self.film_miller, min_slab_size=1, min_vacuum_size=3, in_unit_planes=True, center_slab=True, primitive=True, reorient_lattice=False)
    sub_sg = SlabGenerator(self.substrate_structure, self.substrate_miller, min_slab_size=1, min_vacuum_size=3, in_unit_planes=True, center_slab=True, primitive=True, reorient_lattice=False)
    film_slabs = film_sg.get_slabs()
    sub_slabs = sub_sg.get_slabs()
    film_shifts = [s.shift for s in film_slabs]
    film_terminations = [label_termination(s) for s in film_slabs]
    sub_shifts = [s.shift for s in sub_slabs]
    sub_terminations = [label_termination(s) for s in sub_slabs]
    self._terminations = {(film_label, sub_label): (film_shift, sub_shift) for (film_label, film_shift), (sub_label, sub_shift) in product(zip(film_terminations, film_shifts), zip(sub_terminations, sub_shifts))}
    self.terminations = list(self._terminations)