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
def _find_matches(self) -> None:
    """Finds and stores the ZSL matches."""
    self.zsl_matches = []
    film_sg = SlabGenerator(self.film_structure, self.film_miller, min_slab_size=1, min_vacuum_size=3, in_unit_planes=True, center_slab=True, primitive=True, reorient_lattice=False)
    sub_sg = SlabGenerator(self.substrate_structure, self.substrate_miller, min_slab_size=1, min_vacuum_size=3, in_unit_planes=True, center_slab=True, primitive=True, reorient_lattice=False)
    film_slab = film_sg.get_slab(shift=0)
    sub_slab = sub_sg.get_slab(shift=0)
    film_vectors = film_slab.lattice.matrix
    substrate_vectors = sub_slab.lattice.matrix
    self.zsl_matches = list(self.zslgen(film_vectors[:2], substrate_vectors[:2], lowest=False))
    for match in self.zsl_matches:
        xform = get_2d_transform(film_vectors, match.film_vectors)
        strain, _rot = polar(xform)
        assert_allclose(strain, np.round(strain), atol=1e-12, err_msg='Film lattice vectors changed during ZSL match, check your ZSL Generator parameters')
        xform = get_2d_transform(substrate_vectors, match.substrate_vectors)
        strain, _rot = polar(xform)
        assert_allclose(strain, strain.astype(int), atol=1e-12, err_msg='Substrate lattice vectors changed during ZSL match, check your ZSL Generator parameters')