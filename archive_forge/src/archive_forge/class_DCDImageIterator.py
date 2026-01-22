import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
class DCDImageIterator:
    """"""

    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, indices=-1, ref_atoms=None, aligned=False):
        self.ref_atoms = ref_atoms
        self.aligned = aligned
        if not hasattr(indices, 'start'):
            if indices < 0:
                indices = slice(indices - 1, indices)
            else:
                indices = slice(indices, indices + 1)
        for chunk in self._getslice(fd, indices):
            yield chunk.build()

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd, self.ref_atoms, self.aligned), indices.start, indices.stop, indices.step)
        except ValueError:
            dtype, natoms, nsteps, header_end = _read_metainfo(fd)
            indices_tuple = indices.indices(nsteps + 1)
            iterator = islice(self.ichunks(fd, self.ref_atoms, self.aligned), *indices_tuple)
        return iterator