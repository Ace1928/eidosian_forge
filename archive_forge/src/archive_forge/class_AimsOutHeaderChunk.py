from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@dataclass
class AimsOutHeaderChunk(AimsOutChunk):
    """The header of the aims.out file containing general information."""
    lines: list[str] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def commit_hash(self) -> str:
        """The commit hash for the FHI-aims version."""
        line_start = self.reverse_search_for(['Commit number'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('This file does not appear to be an aims-output file')
        return self.lines[line_start].split(':')[1].strip()

    @property
    def aims_uuid(self) -> str:
        """The aims-uuid for the calculation."""
        line_start = self.reverse_search_for(['aims_uuid'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('This file does not appear to be an aims-output file')
        return self.lines[line_start].split(':')[1].strip()

    @property
    def version_number(self) -> str:
        """The commit hash for the FHI-aims version."""
        line_start = self.reverse_search_for(['FHI-aims version'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('This file does not appear to be an aims-output file')
        return self.lines[line_start].split(':')[1].strip()

    @property
    def fortran_compiler(self) -> str | None:
        """The fortran compiler used to make FHI-aims."""
        line_start = self.reverse_search_for(['Fortran compiler      :'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('This file does not appear to be an aims-output file')
        return self.lines[line_start].split(':')[1].split('/')[-1].strip()

    @property
    def c_compiler(self) -> str | None:
        """The C compiler used to make FHI-aims."""
        line_start = self.reverse_search_for(['C compiler            :'])
        if line_start == LINE_NOT_FOUND:
            return None
        return self.lines[line_start].split(':')[1].split('/')[-1].strip()

    @property
    def fortran_compiler_flags(self) -> str | None:
        """The fortran compiler flags used to make FHI-aims."""
        line_start = self.reverse_search_for(['Fortran compiler flags'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('This file does not appear to be an aims-output file')
        return self.lines[line_start].split(':')[1].strip()

    @property
    def c_compiler_flags(self) -> str | None:
        """The C compiler flags used to make FHI-aims."""
        line_start = self.reverse_search_for(['C compiler flags'])
        if line_start == LINE_NOT_FOUND:
            return None
        return self.lines[line_start].split(':')[1].strip()

    @property
    def build_type(self) -> list[str]:
        """The optional build flags passed to cmake."""
        line_end = self.reverse_search_for(['Linking against:'])
        line_inds = self.search_for_all('Using', line_end=line_end)
        return [' '.join(self.lines[ind].split()[1:]).strip() for ind in line_inds]

    @property
    def linked_against(self) -> list[str]:
        """Get all libraries used to link the FHI-aims executable."""
        line_start = self.reverse_search_for(['Linking against:'])
        if line_start == LINE_NOT_FOUND:
            return []
        linked_libs = [self.lines[line_start].split(':')[1].strip()]
        line_start += 1
        while 'lib' in self.lines[line_start]:
            linked_libs.append(self.lines[line_start].strip())
            line_start += 1
        return linked_libs

    @property
    def initial_lattice(self) -> Lattice | None:
        """The initial lattice vectors from the aims.out file."""
        line_start = self.reverse_search_for(['| Unit cell:'])
        if line_start == LINE_NOT_FOUND:
            return None
        return Lattice(np.array([[float(inp) for inp in line.split()[-3:]] for line in self.lines[line_start + 1:line_start + 4]]))

    @property
    def initial_structure(self) -> Structure | Molecule:
        """The initial structure

        Using the FHI-aims output file recreate the initial structure for
        the calculation.
        """
        lattice = self.initial_lattice
        line_start = self.reverse_search_for(['Atomic structure:'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('No information about the structure in the chunk.')
        line_start += 2
        coords = np.zeros((self.n_atoms, 3))
        species = [''] * self.n_atoms
        for ll, line in enumerate(self.lines[line_start:line_start + self.n_atoms]):
            inp = line.split()
            coords[ll, :] = [float(pos) for pos in inp[4:7]]
            species[ll] = inp[3]
        site_properties = {'charge': self.initial_charges}
        if self.initial_magnetic_moments is not None:
            site_properties['magmoms'] = self.initial_magnetic_moments
        if lattice:
            return Structure(lattice, species, coords, np.sum(self.initial_charges), coords_are_cartesian=True, site_properties=site_properties)
        return Molecule(species, coords, np.sum(self.initial_charges), site_properties=site_properties)

    @property
    def initial_charges(self) -> Sequence[float]:
        """The initial charges for the structure"""
        if 'initial_charges' not in self._cache:
            self._parse_initial_charges_and_moments()
        return self._cache['initial_charges']

    @property
    def initial_magnetic_moments(self) -> Sequence[float]:
        """The initial magnetic Moments"""
        if 'initial_magnetic_moments' not in self._cache:
            self._parse_initial_charges_and_moments()
        return self._cache['initial_magnetic_moments']

    def _parse_initial_charges_and_moments(self) -> None:
        """Parse the initial charges and magnetic moments from a file"""
        charges = np.zeros(self.n_atoms)
        magmoms = None
        line_start = self.reverse_search_for(['Initial charges', 'Initial moments and charges'])
        if line_start != LINE_NOT_FOUND:
            line_start += 2
            magmoms = np.zeros(self.n_atoms)
            for ll, line in enumerate(self.lines[line_start:line_start + self.n_atoms]):
                inp = line.split()
                if len(inp) == 4:
                    charges[ll] = float(inp[2])
                    magmoms = None
                else:
                    charges[ll] = float(inp[3])
                    magmoms[ll] = float(inp[2])
        self._cache['initial_charges'] = charges
        self._cache['initial_magnetic_moments'] = magmoms

    @property
    def is_md(self) -> bool:
        """Is the output for a molecular dynamics calculation?"""
        return self.reverse_search_for(['Complete information for previous time-step:']) != LINE_NOT_FOUND

    @property
    def is_relaxation(self) -> bool:
        """Is the output for a relaxation?"""
        return self.reverse_search_for(['Geometry relaxation:']) != LINE_NOT_FOUND

    def _parse_k_points(self) -> None:
        """Parse the list of k-points used in the calculation."""
        n_kpts = self.parse_scalar('n_kpts')
        if n_kpts is None:
            self._cache.update({'k_points': None, 'k_point_weights': None})
            return
        n_kpts = int(n_kpts)
        line_start = self.reverse_search_for(['| K-points in task'])
        line_end = self.reverse_search_for(['| k-point:'])
        if line_start == LINE_NOT_FOUND or line_end == LINE_NOT_FOUND or line_end - line_start != n_kpts:
            self._cache.update({'k_points': None, 'k_point_weights': None})
            return
        k_points = np.zeros((n_kpts, 3))
        k_point_weights = np.zeros(n_kpts)
        for kk, line in enumerate(self.lines[line_start + 1:line_end + 1]):
            k_points[kk] = [float(inp) for inp in line.split()[4:7]]
            k_point_weights[kk] = float(line.split()[-1])
        self._cache.update({'k_points': k_points, 'k_point_weights': k_point_weights})

    @property
    def n_atoms(self) -> int:
        """The number of atoms for the material."""
        n_atoms = self.parse_scalar('n_atoms')
        if n_atoms is None:
            raise AimsParseError('No information about the number of atoms in the header.')
        return int(n_atoms)

    @property
    def n_bands(self) -> int | None:
        """The number of Kohn-Sham states for the chunk."""
        line_start = self.reverse_search_for(SCALAR_PROPERTY_TO_LINE_KEY['n_bands'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('No information about the number of Kohn-Sham states in the header.')
        line = self.lines[line_start]
        if '| Number of Kohn-Sham states' in line:
            return int(line.split(':')[-1].strip().split()[0])
        return int(line.split()[-1].strip()[:-1])

    @property
    def n_electrons(self) -> int | None:
        """The number of electrons for the chunk."""
        line_start = self.reverse_search_for(SCALAR_PROPERTY_TO_LINE_KEY['n_electrons'])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError('No information about the number of electrons in the header.')
        line = self.lines[line_start]
        return int(float(line.split()[-2]))

    @property
    def n_k_points(self) -> int | None:
        """The number of k_ppoints for the calculation."""
        n_kpts = self.parse_scalar('n_kpts')
        if n_kpts is None:
            return None
        return int(n_kpts)

    @property
    def n_spins(self) -> int | None:
        """The number of spin channels for the chunk."""
        n_spins = self.parse_scalar('n_spins')
        if n_spins is None:
            raise AimsParseError('No information about the number of spin channels in the header.')
        return int(n_spins)

    @property
    def electronic_temperature(self) -> float:
        """The electronic temperature for the chunk."""
        line_start = self.reverse_search_for(SCALAR_PROPERTY_TO_LINE_KEY['electronic_temp'])
        if line_start == LINE_NOT_FOUND:
            return 0.0
        line = self.lines[line_start]
        return float(line.split('=')[-1].strip().split()[0])

    @property
    def k_points(self) -> Sequence[Vector3D]:
        """All k-points listed in the calculation."""
        if 'k_points' not in self._cache:
            self._parse_k_points()
        return self._cache['k_points']

    @property
    def k_point_weights(self) -> Sequence[float]:
        """The k-point weights for the calculation."""
        if 'k_point_weights' not in self._cache:
            self._parse_k_points()
        return self._cache['k_point_weights']

    @property
    def header_summary(self) -> dict[str, Any]:
        """Dictionary summarizing the information inside the header."""
        return {'initial_structure': self.initial_structure, 'initial_lattice': self.initial_lattice, 'is_relaxation': self.is_relaxation, 'is_md': self.is_md, 'n_atoms': self.n_atoms, 'n_bands': self.n_bands, 'n_electrons': self.n_electrons, 'n_spins': self.n_spins, 'electronic_temperature': self.electronic_temperature, 'n_k_points': self.n_k_points, 'k_points': self.k_points, 'k_point_weights': self.k_point_weights}

    @property
    def metadata_summary(self) -> dict[str, list[str] | str | None]:
        """Dictionary containing all metadata for FHI-aims build."""
        return {'commit_hash': self.commit_hash, 'aims_uuid': self.aims_uuid, 'version_number': self.version_number, 'fortran_compiler': self.fortran_compiler, 'c_compiler': self.c_compiler, 'fortran_compiler_flags': self.fortran_compiler_flags, 'c_compiler_flags': self.c_compiler_flags, 'build_type': self.build_type, 'linked_against': self.linked_against}