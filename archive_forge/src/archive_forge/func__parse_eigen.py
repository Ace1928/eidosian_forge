from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
def _parse_eigen(self) -> np.ndarray:
    """Parse REPORT file to obtain information about eigenvalues.

        Returns:
            np.ndarray: Eigenvalues with shape of (1 or 2, n_kpoints, n_bands).
                The first index represents spin, the second index represents
                kpoints, the third index represents band.
        """
    num_rows: int = int(np.ceil(self._num_bands / 5))
    content: str = 'eigen energies, in eV'
    rows_lst: list[int] = LineLocator.locate_all_lines(file_path=self.filename, content=content)
    rows_array: np.ndarray = np.array(rows_lst).reshape(self._spin, -1)
    eigenvalues: np.ndarray = np.zeros((self._spin, self._num_kpts, self._num_bands))
    for ii in range(self._spin):
        for jj in range(self._num_kpts):
            tmp_eigenvalues_str = ''
            for kk in range(num_rows):
                tmp_eigenvalues_str += linecache.getline(str(self.filename), rows_array[ii][jj] + kk + 1)
            tmp_eigenvalues_array = np.array([float(eigen_value) for eigen_value in tmp_eigenvalues_str.split()])
            for kk in range(self._num_bands):
                eigenvalues[ii][jj][kk] = tmp_eigenvalues_array[kk]
    return eigenvalues