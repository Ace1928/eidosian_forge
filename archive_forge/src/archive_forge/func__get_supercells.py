from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
def _get_supercells(self, struct1, struct2, fu, s1_supercell):
    """
        Computes all supercells of one structure close to the lattice of the
        other
        if s1_supercell is True, it makes the supercells of struct1, otherwise
        it makes them of s2.

        yields: s1, s2, supercell_matrix, average_lattice, supercell_matrix
        """

    def av_lat(l1, l2):
        params = (np.array(l1.parameters) + np.array(l2.parameters)) / 2
        return Lattice.from_parameters(*params)

    def sc_generator(s1, s2):
        s2_fc = np.array(s2.frac_coords)
        if fu == 1:
            cc = np.array(s1.cart_coords)
            for latt, sc_m in self._get_lattices(s2.lattice, s1, fu):
                fc = latt.get_fractional_coords(cc)
                fc -= np.floor(fc)
                yield (fc, s2_fc, av_lat(latt, s2.lattice), sc_m)
        else:
            fc_init = np.array(s1.frac_coords)
            for latt, sc_m in self._get_lattices(s2.lattice, s1, fu):
                fc = np.dot(fc_init, np.linalg.inv(sc_m))
                lp = lattice_points_in_supercell(sc_m)
                fc = (fc[:, None, :] + lp[None, :, :]).reshape((-1, 3))
                fc -= np.floor(fc)
                yield (fc, s2_fc, av_lat(latt, s2.lattice), sc_m)
    if s1_supercell:
        for x in sc_generator(struct1, struct2):
            yield x
    else:
        for x in sc_generator(struct2, struct1):
            yield (x[1], x[0], x[2], x[3])