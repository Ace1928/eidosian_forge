import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
class ElkReader:

    def __init__(self, path):
        self.path = Path(path)

    def _read_everything(self):
        yield from self._read_energy()
        with (self.path / 'INFO.OUT').open() as fd:
            yield from parse_elk_info(fd)
        with (self.path / 'EIGVAL.OUT').open() as fd:
            yield from parse_elk_eigval(fd)
        with (self.path / 'KPOINTS.OUT').open() as fd:
            yield from parse_elk_kpoints(fd)

    def read_everything(self):
        dct = dict(self._read_everything())
        spinpol = dct.pop('spinpol')
        if spinpol:
            for name in ('eigenvalues', 'occupations'):
                array = dct[name]
                _, nkpts, nbands_double = array.shape
                assert _ == 1
                assert nbands_double % 2 == 0
                nbands = nbands_double // 2
                newarray = np.empty((2, nkpts, nbands))
                newarray[0, :, :] = array[0, :, :nbands]
                newarray[1, :, :] = array[0, :, nbands:]
                if name == 'eigenvalues':
                    diffs = np.diff(newarray, axis=2)
                    assert all(diffs.flat[:] > 0)
                dct[name] = newarray
        return dct

    def _read_energy(self):
        txt = (self.path / 'TOTENERGY.OUT').read_text()
        tokens = txt.split()
        energy = float(tokens[-1]) * Hartree
        yield ('free_energy', energy)
        yield ('energy', energy)