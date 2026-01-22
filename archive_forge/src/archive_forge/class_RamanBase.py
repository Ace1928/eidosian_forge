import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
class RamanBase(AtomicDisplacements, IOContext):

    def __init__(self, atoms, *args, name='raman', exname=None, exext='.alpha', txt='-', verbose=False, comm=world, **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exext: string
          Extension for excitation filenames
        txt:
          Output stream
        verbose:
          Verbosity level of output
        comm:
          Communicator, default world
        """
        self.atoms = atoms
        self.name = name
        if exname is None:
            self.exname = name
        else:
            self.exname = exname
        self.exext = exext
        self.txt = self.openfile(txt, comm)
        self.verbose = verbose
        self.comm = comm