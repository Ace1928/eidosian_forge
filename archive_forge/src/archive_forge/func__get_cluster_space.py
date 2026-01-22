from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def _get_cluster_space(self) -> ClusterSpace:
    """Generate the ClusterSpace object for icet."""
    chemical_symbols = [list(site.species.as_dict()) for site in self._structure]
    return ClusterSpace(structure=self._ordered_atoms, cutoffs=self.cutoffs_list, chemical_symbols=chemical_symbols)