from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
class Movement(MSONable):
    """Parser for data in MOVEMENT which records trajectory during MD."""

    def __init__(self, filename: PathLike, ionic_step_skip: int | None=None, ionic_step_offset: int | None=None):
        """Initialization function.

        Args:
            filename (PathLike): The path of MOVEMENT
            ionic_step_skip (int | None, optional): If ionic_step_skip is a number > 1,
                only every ionic_step_skip ionic steps will be read for
                structure and energies. This is very useful if you are parsing
                very large MOVEMENT files. Defaults to None.
            ionic_step_offset (int | None, optional): Used together with ionic_step_skip.
                If set, the first ionic step read will be offset by the amount of
                ionic_step_offset. Defaults to None.
        """
        self.filename: PathLike = filename
        self.ionic_step_skip: int | None = ionic_step_skip
        self.ionic_step_offset: int | None = ionic_step_offset
        self.split_mark: str = '--------------------------------------'
        self.chunk_sizes, self.chunk_starts = self._get_chunk_info()
        self.n_ionic_steps: int = len(self.chunk_sizes)
        self.ionic_steps: list[dict] = self._parse_sefv()
        if self.ionic_step_offset and self.ionic_step_skip:
            self.ionic_steps = self.ionic_steps[self.ionic_step_offset::self.ionic_step_skip]

    def _get_chunk_info(self) -> tuple[list[int], list[int]]:
        """Split MOVEMENT into many chunks, so that program process it chunk by chunk.

        Returns:
            tuple[list[int], list[int]]:
                chunk_sizes (list[int]): The number of lines occupied by structural
                    information in each step.
                chunk_starts (list[int]): The starting line number for structural
                    information in each step.
        """
        chunk_sizes: list[int] = []
        row_idxs: list[int] = LineLocator.locate_all_lines(self.filename, self.split_mark)
        chunk_sizes.append(row_idxs[0])
        for ii in range(1, len(row_idxs)):
            chunk_sizes.append(row_idxs[ii] - row_idxs[ii - 1])
        chunk_sizes_bak: list[int] = copy.deepcopy(chunk_sizes)
        chunk_sizes_bak.insert(0, 0)
        chunk_starts: list[int] = np.cumsum(chunk_sizes_bak).tolist()
        chunk_starts.pop(-1)
        return (chunk_sizes, chunk_starts)

    @property
    def atom_configs(self) -> list[Structure]:
        """Returns AtomConfig object for structures contained in MOVEMENT.

        Returns:
            list[Structure]: List of Structure objects for the structure at each ionic step.
        """
        return [step['atom_config'] for step in self.ionic_steps]

    @property
    def e_tots(self) -> np.ndarray:
        """Returns total energies of each ionic step structures contained in MOVEMENT.

        Returns:
            np.ndarray: Total energy of of each ionic step structure,
                with shape of (n_ionic_steps,).
        """
        return np.array([step['e_tot'] for step in self.ionic_steps])

    @property
    def atom_forces(self) -> np.ndarray:
        """Returns forces on atoms in each structures contained in MOVEMENT.

        Returns:
            np.ndarray: The forces on atoms of each ionic step structure,
                with shape of (n_ionic_steps, n_atoms, 3).
        """
        return np.array([step['atom_forces'] for step in self.ionic_steps])

    @property
    def e_atoms(self) -> np.ndarray:
        """
        Returns individual energies of atoms in each ionic step structures
        contained in MOVEMENT.

        Returns:
            np.ndarray: The individual energy of atoms in each ionic step structure,
                with shape of (n_ionic_steps, n_atoms).
        """
        return np.array([step['eatoms'] for step in self.ionic_steps if 'eatoms' in step])

    @property
    def virials(self) -> np.ndarray:
        """Returns virial tensor of each ionic step structure contained in MOVEMENT.

        Returns:
            np.ndarray: The virial tensor of each ionic step structure,
                with shape of (n_ionic_steps, 3, 3)
        """
        return np.array([step['virial'] for step in self.ionic_steps if 'virial' in step])

    def _parse_sefv(self) -> list[dict]:
        """
        Parse the MOVEMENT file, return information ionic step structure containing
        structures, energies, forces on atoms and virial tensor.

        Returns:
            list[dict]: Structure containing structures, energies, forces on atoms
                and virial tensor. The corresponding keys are 'atom_config', 'e_tot',
                'atom_forces' and 'virial'.
        """
        ionic_steps: list[dict] = []
        with zopen(self.filename, 'rt') as mvt:
            tmp_step: dict = {}
            for ii in range(self.n_ionic_steps):
                tmp_chunk: str = ''
                for _ in range(self.chunk_sizes[ii]):
                    tmp_chunk += mvt.readline()
                tmp_step.update({'atom_config': AtomConfig.from_str(tmp_chunk)})
                tmp_step.update({'e_tot': ACstrExtractor(tmp_chunk).get_e_tot()[0]})
                tmp_step.update({'atom_forces': ACstrExtractor(tmp_chunk).get_atom_forces().reshape(-1, 3)})
                e_atoms: np.ndarray | None = ACstrExtractor(tmp_chunk).get_atom_forces()
                if e_atoms is not None:
                    tmp_step.update({'atom_energies': ACstrExtractor(tmp_chunk).get_atom_energies()})
                else:
                    print(f'Ionic step #{ii} : Energy deposition is turn down.')
                virial: np.ndarray | None = ACstrExtractor(tmp_chunk).get_virial()
                if virial is not None:
                    tmp_step.update({'virial': virial.reshape(3, 3)})
                else:
                    print(f'Ionic step #{ii} : No virial information.')
                ionic_steps.append(tmp_step)
        return ionic_steps