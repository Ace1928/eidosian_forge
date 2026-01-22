from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def as_phononwebsite(self) -> dict:
    """Return a dictionary with the phononwebsite format:
        http://henriquemiranda.github.io/phononwebsite.
        """
    assert self.structure is not None, 'Structure is required for as_phononwebsite'
    dct = {}
    dct['lattice'] = self.structure.lattice._matrix.tolist()
    atom_pos_car = []
    atom_pos_red = []
    atom_types = []
    for site in self.structure:
        atom_pos_car.append(site.coords.tolist())
        atom_pos_red.append(site.frac_coords.tolist())
        atom_types.append(site.species_string)
    dct['repetitions'] = get_reasonable_repetitions(len(atom_pos_car))
    dct['natoms'] = len(atom_pos_car)
    dct['atom_pos_car'] = atom_pos_car
    dct['atom_pos_red'] = atom_pos_red
    dct['atom_types'] = atom_types
    dct['atom_numbers'] = self.structure.atomic_numbers
    dct['formula'] = self.structure.formula
    dct['name'] = self.structure.formula
    qpoints = []
    for q_pt in self.qpoints:
        qpoints.append(list(q_pt.frac_coords))
    dct['qpoints'] = qpoints
    hsq_dict = {}
    for nq, q_pt in enumerate(self.qpoints):
        if q_pt.label is not None:
            hsq_dict[nq] = q_pt.label
    dist = 0
    nq_start = 0
    distances = [dist]
    line_breaks = []
    for nq in range(1, len(qpoints)):
        q1 = np.array(qpoints[nq])
        q2 = np.array(qpoints[nq - 1])
        if nq in hsq_dict and nq - 1 in hsq_dict:
            if hsq_dict[nq] != hsq_dict[nq - 1]:
                hsq_dict[nq - 1] += '|' + hsq_dict[nq]
            del hsq_dict[nq]
            line_breaks.append((nq_start, nq))
            nq_start = nq
        else:
            dist += np.linalg.norm(q1 - q2)
        distances.append(dist)
    line_breaks.append((nq_start, len(qpoints)))
    dct['distances'] = distances
    dct['line_breaks'] = line_breaks
    dct['highsym_qpts'] = list(hsq_dict.items())
    thz2cm1 = 33.35641
    bands = self.bands.copy() * thz2cm1
    dct['eigenvalues'] = bands.T.tolist()
    eigen_vecs = self.eigendisplacements.copy()
    eigen_vecs /= np.linalg.norm(eigen_vecs[0, 0])
    eigen_vecs = eigen_vecs.swapaxes(0, 1)
    eigen_vecs = np.array([eigen_vecs.real, eigen_vecs.imag])
    eigen_vecs = np.rollaxis(eigen_vecs, 0, 5)
    dct['vectors'] = eigen_vecs.tolist()
    return dct