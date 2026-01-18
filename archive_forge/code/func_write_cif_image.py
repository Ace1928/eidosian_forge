import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
def write_cif_image(blockname, atoms, fd, *, wrap, labels, loop_keys):
    fd.write(blockname)
    fd.write(chemical_formula_header(atoms))
    rank = atoms.cell.rank
    if rank == 3:
        fd.write(format_cell(atoms.cell))
        fd.write('\n')
        fd.write(format_generic_spacegroup_info())
        fd.write('\n')
    elif rank != 0:
        raise ValueError(f'CIF format can only represent systems with 0 or 3 lattice vectors.  Got {rank}.')
    loopdata, coord_headers = atoms_to_loop_data(atoms, wrap, labels, loop_keys)
    headers = ['_atom_site_type_symbol', '_atom_site_label', '_atom_site_symmetry_multiplicity', *coord_headers, '_atom_site_occupancy']
    headers += ['_' + key for key in loop_keys]
    loop = CIFLoop()
    for header in headers:
        array, fmt = loopdata[header]
        loop.add(header, array, fmt)
    fd.write(loop.tostring())