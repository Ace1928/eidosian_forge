from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
class PhononBandStructureSymmLine(PhononBandStructure):
    """This object stores phonon band structures along selected (symmetry) lines in the
    Brillouin zone. We call the different symmetry lines (ex: \\\\Gamma to Z)
    "branches".
    """

    def __init__(self, qpoints: Sequence[Kpoint], frequencies: ArrayLike, lattice: Lattice, has_nac: bool=False, eigendisplacements: ArrayLike=None, labels_dict: dict | None=None, coords_are_cartesian: bool=False, structure: Structure | None=None) -> None:
        """
        Args:
            qpoints: list of qpoints as numpy arrays, in frac_coords of the
                given lattice by default
            frequencies: list of phonon frequencies in eV as a numpy array with shape
                (3*len(structure), len(qpoints))
            lattice: The reciprocal lattice as a pymatgen Lattice object.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient
            has_nac: specify if the band structure has been produced taking into account
                non-analytical corrections at Gamma. If True frequencies at Gamma from
                different directions will be stored in naf. Default False.
            eigendisplacements: the phonon eigendisplacements associated to the
                frequencies in Cartesian coordinates. A numpy array of complex
                numbers with shape (3*len(structure), len(qpoints), len(structure), 3).
                he First index of the array refers to the band, the second to the index
                of the qpoint, the third to the atom in the structure and the fourth
                to the Cartesian coordinates.
            labels_dict: (dict) of {} this links a qpoint (in frac coords or
                Cartesian coordinates depending on the coords) to a label.
            coords_are_cartesian: Whether the qpoint coordinates are cartesian.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed if we
                provide projections to the band structure.
        """
        super().__init__(qpoints=qpoints, frequencies=frequencies, lattice=lattice, nac_frequencies=None, eigendisplacements=eigendisplacements, nac_eigendisplacements=None, labels_dict=labels_dict, coords_are_cartesian=coords_are_cartesian, structure=structure)
        self._reuse_init(eigendisplacements, frequencies, has_nac, qpoints)

    def __repr__(self) -> str:
        bands, labels = (self.bands.shape, list(self.labels_dict))
        return f'{type(self).__name__}(bands={bands!r}, labels={labels!r})'

    def _reuse_init(self, eigendisplacements: ArrayLike, frequencies: ArrayLike, has_nac: bool, qpoints: Sequence[Kpoint]) -> None:
        self.distance = []
        self.branches = []
        one_group: list = []
        branches_tmp = []
        previous_qpoint = self.qpoints[0]
        previous_distance = 0.0
        previous_label = self.qpoints[0].label
        for idx in range(self.nb_qpoints):
            label = self.qpoints[idx].label
            if label is not None and previous_label is not None:
                self.distance += [previous_distance]
            else:
                self.distance += [np.linalg.norm(self.qpoints[idx].cart_coords - previous_qpoint.cart_coords) + previous_distance]
            previous_qpoint = self.qpoints[idx]
            previous_distance = self.distance[idx]
            if label and previous_label:
                if len(one_group) != 0:
                    branches_tmp += [one_group]
                one_group = []
            previous_label = label
            one_group += [idx]
        if len(one_group) != 0:
            branches_tmp += [one_group]
        for branch in branches_tmp:
            self.branches += [{'start_index': branch[0], 'end_index': branch[-1], 'name': f'{self.qpoints[branch[0]].label}-{self.qpoints[branch[-1]].label}'}]
        if has_nac:
            naf = []
            nac_eigendisplacements = []
            for idx in range(self.nb_qpoints):
                if np.allclose(qpoints[idx], (0, 0, 0)):
                    if idx > 0 and (not np.allclose(qpoints[idx - 1], (0, 0, 0))):
                        q_dir = self.qpoints[idx - 1]
                        direction = q_dir.frac_coords / np.linalg.norm(q_dir.frac_coords)
                        naf.append((direction, frequencies[:, idx]))
                        if self.has_eigendisplacements:
                            nac_eigendisplacements.append((direction, eigendisplacements[:, idx]))
                    if idx < len(qpoints) - 1 and (not np.allclose(qpoints[idx + 1], (0, 0, 0))):
                        q_dir = self.qpoints[idx + 1]
                        direction = q_dir.frac_coords / np.linalg.norm(q_dir.frac_coords)
                        naf.append((direction, frequencies[:, idx]))
                        if self.has_eigendisplacements:
                            nac_eigendisplacements.append((direction, eigendisplacements[:, idx]))
            self.nac_frequencies = np.array(naf, dtype=object)
            self.nac_eigendisplacements = np.array(nac_eigendisplacements, dtype=object)

    def get_equivalent_qpoints(self, index: int) -> list[int]:
        """Returns the list of qpoint indices equivalent (meaning they are the
        same frac coords) to the given one.

        Args:
            index (int): the qpoint index

        Returns:
            list[int]: equivalent indices

        TODO: now it uses the label we might want to use coordinates instead
        (in case there was a mislabel)
        """
        if self.qpoints[index].label is None:
            return [index]
        list_index_qpoints = []
        for idx in range(self.nb_qpoints):
            if self.qpoints[idx].label == self.qpoints[index].label:
                list_index_qpoints.append(idx)
        return list_index_qpoints

    def get_branch(self, index: int) -> list[dict[str, str | int]]:
        """Returns in what branch(es) is the qpoint. There can be several branches.

        Args:
            index (int): the qpoint index

        Returns:
            list[dict[str, str | int]]: [{"name","start_index","end_index","index"}]
                indicating all branches in which the qpoint is. It takes into
                account the fact that one qpoint (e.g., \\\\Gamma) can be in several
                branches
        """
        lst = []
        for pt_idx in self.get_equivalent_qpoints(index):
            for branch in self.branches:
                start_idx, end_idx = (branch['start_index'], branch['end_index'])
                if start_idx <= pt_idx <= end_idx:
                    lst.append({'name': branch['name'], 'start_index': start_idx, 'end_index': end_idx, 'index': pt_idx})
        return lst

    def write_phononwebsite(self, filename: str | PathLike) -> None:
        """Write a json file for the phononwebsite:
        http://henriquemiranda.github.io/phononwebsite.
        """
        with open(filename, mode='w') as file:
            json.dump(self.as_phononwebsite(), file)

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

    def band_reorder(self) -> None:
        """Re-order the eigenvalues according to the similarity of the eigenvectors."""
        eigen_displacements = self.eigendisplacements
        eig = self.bands
        n_phonons, n_qpoints = self.bands.shape
        order = np.zeros([n_qpoints, n_phonons], dtype=int)
        order[0] = np.array(range(n_phonons))
        assert self.structure is not None, 'Structure is required for band_reorder'
        atomic_masses = [site.specie.atomic_mass for site in self.structure]
        for nq in range(1, n_qpoints):
            old_eig_vecs = eigenvectors_from_displacements(eigen_displacements[:, nq - 1], atomic_masses)
            new_eig_vecs = eigenvectors_from_displacements(eigen_displacements[:, nq], atomic_masses)
            order[nq] = estimate_band_connection(old_eig_vecs.reshape([n_phonons, n_phonons]).T, new_eig_vecs.reshape([n_phonons, n_phonons]).T, order[nq - 1])
        for nq in range(1, n_qpoints):
            eivq = eigen_displacements[:, nq]
            eigq = eig[:, nq]
            eigen_displacements[:, nq] = eivq[order[nq]]
            eig[:, nq] = eigq[order[nq]]

    def as_dict(self) -> dict:
        """Returns: MSONable dict."""
        dct = super().as_dict()
        nac_frequencies = dct.pop('nac_frequencies')
        dct.pop('nac_eigendisplacements')
        dct['has_nac'] = len(nac_frequencies) > 0
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            PhononBandStructureSymmLine
        """
        lattice_rec = Lattice(dct['lattice_rec']['matrix'])
        eigendisplacements = np.array(dct['eigendisplacements']['real']) + np.array(dct['eigendisplacements']['imag']) * 1j
        return cls(dct['qpoints'], np.array(dct['bands']), lattice_rec, dct['has_nac'], eigendisplacements, dct['labels_dict'], structure=Structure.from_dict(dct['structure']) if 'structure' in dct else None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhononBandStructureSymmLine):
            return NotImplemented
        return self.bands.shape == other.bands.shape and np.allclose(self.bands, other.bands) and (self.lattice_rec == other.lattice_rec) and (self.labels_dict == other.labels_dict) and (self.structure == other.structure)