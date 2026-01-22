from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.dev import requires
from monty.json import MSONable
from scipy.interpolate import UnivariateSpline
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import amu_to_kg
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
class GruneisenPhononBandStructureSymmLine(GruneisenPhononBandStructure, PhononBandStructureSymmLine):
    """This object stores a GruneisenPhononBandStructureSymmLine together with Grueneisen parameters
    for every frequency.
    """

    def __init__(self, qpoints: ArrayLike, frequencies: ArrayLike[ArrayLike], gruneisenparameters: ArrayLike, lattice: Lattice, eigendisplacements: ArrayLike[ArrayLike]=None, labels_dict: dict | None=None, coords_are_cartesian: bool=False, structure: Structure | None=None) -> None:
        """
        Args:
            qpoints: list of qpoints as numpy arrays, in frac_coords of the
                given lattice by default
            frequencies: list of phonon frequencies in eV as a numpy array with shape
                (3*len(structure), len(qpoints))
            gruneisenparameters: list of Grueneisen parameters as a numpy array with the
                shape (3*len(structure), len(qpoints))
            lattice: The reciprocal lattice as a pymatgen Lattice object.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient
            eigendisplacements: the phonon eigendisplacements associated to the
                frequencies in Cartesian coordinates. A numpy array of complex
                numbers with shape (3*len(structure), len(qpoints), len(structure), 3).
                The first index of the array refers to the band, the second to the index
                of the qpoint, the third to the atom in the structure and the fourth
                to the Cartesian coordinates.
            labels_dict: (dict) of {} this links a qpoint (in frac coords or
                Cartesian coordinates depending on the coords) to a label.
            coords_are_cartesian: Whether the qpoint coordinates are cartesian.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed if we
                provide projections to the band structure.
        """
        GruneisenPhononBandStructure.__init__(self, qpoints=qpoints, frequencies=frequencies, gruneisenparameters=gruneisenparameters, lattice=lattice, eigendisplacements=eigendisplacements, labels_dict=labels_dict, coords_are_cartesian=coords_are_cartesian, structure=structure)
        PhononBandStructureSymmLine._reuse_init(self, eigendisplacements=eigendisplacements, frequencies=frequencies, has_nac=False, qpoints=qpoints)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            GruneisenPhononBandStructureSymmLine
        """
        lattice_rec = Lattice(dct['lattice_rec']['matrix'])
        eigendisplacements = np.array(dct['eigendisplacements']['real']) + np.array(dct['eigendisplacements']['imag']) * 1j
        structure = Structure.from_dict(dct['structure']) if 'structure' in dct else None
        return cls(qpoints=dct['qpoints'], frequencies=np.array(dct['bands']), gruneisenparameters=np.array(dct['gruneisen']), lattice=lattice_rec, eigendisplacements=eigendisplacements, labels_dict=dct['labels_dict'], structure=structure)