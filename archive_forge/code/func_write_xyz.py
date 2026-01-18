from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
def write_xyz(fileobj, images, comment='', columns=None, write_info=True, write_results=True, plain=False, vec_cell=False, append=False):
    """
    Write output in extended XYZ format

    Optionally, specify which columns (arrays) to include in output,
    whether to write the contents of the `atoms.info` dict to the
    XYZ comment line (default is True), the results of any
    calculator attached to this Atoms. The `plain` argument
    can be used to write a simple XYZ file with no additional information.
    `vec_cell` can be used to write the cell vectors as additional
    pseudo-atoms. If `append` is set to True, the file is for append (mode `a`),
    otherwise it is overwritten (mode `w`).

    See documentation for :func:`read_xyz()` for further details of the extended
    XYZ file format.
    """
    if isinstance(fileobj, str):
        mode = 'w'
        if append:
            mode = 'a'
        fileobj = paropen(fileobj, mode)
    if hasattr(images, 'get_positions'):
        images = [images]
    for atoms in images:
        natoms = len(atoms)
        if columns is None:
            fr_cols = None
        else:
            fr_cols = columns[:]
        if fr_cols is None:
            fr_cols = ['symbols', 'positions'] + [key for key in atoms.arrays.keys() if key not in ['symbols', 'positions', 'numbers', 'species', 'pos']]
        if vec_cell:
            plain = True
        if plain:
            fr_cols = ['symbols', 'positions']
            write_info = False
            write_results = False
        per_frame_results = {}
        per_atom_results = {}
        if write_results:
            calculator = atoms.calc
            if calculator is not None and isinstance(calculator, Calculator):
                for key in all_properties:
                    value = calculator.results.get(key, None)
                    if value is None:
                        continue
                    if key in per_atom_properties and len(value.shape) >= 1 and (value.shape[0] == len(atoms)):
                        per_atom_results[key] = value
                    elif key in per_config_properties:
                        if key == 'stress':
                            xx, yy, zz, yz, xz, xy = value
                            value = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
                        per_frame_results[key] = value
        if 'symbols' in fr_cols:
            i = fr_cols.index('symbols')
            fr_cols[0], fr_cols[i] = (fr_cols[i], fr_cols[0])
        if 'positions' in fr_cols:
            i = fr_cols.index('positions')
            fr_cols[1], fr_cols[i] = (fr_cols[i], fr_cols[1])
        if fr_cols[0] in atoms.arrays:
            symbols = atoms.arrays[fr_cols[0]]
        else:
            symbols = atoms.get_chemical_symbols()
        if natoms > 0 and (not isinstance(symbols[0], str)):
            raise ValueError('First column must be symbols-like')
        pos = atoms.arrays[fr_cols[1]]
        if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
            raise ValueError('Second column must be position-like')
        if vec_cell:
            pbc = list(atoms.get_pbc())
            cell = atoms.get_cell()
            if True in pbc:
                nPBC = 0
                for i, b in enumerate(pbc):
                    if b:
                        nPBC += 1
                        symbols.append('VEC' + str(nPBC))
                        pos = np.vstack((pos, cell[i]))
                natoms += nPBC
                if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
                    raise ValueError('Pseudo Atoms containing cell have bad coords')
        if 'move_mask' in fr_cols:
            cnstr = images[0]._get_constraints()
            if len(cnstr) > 0:
                c0 = cnstr[0]
                if isinstance(c0, FixAtoms):
                    cnstr = np.ones((natoms,), dtype=bool)
                    for idx in c0.index:
                        cnstr[idx] = False
                elif isinstance(c0, FixCartesian):
                    masks = np.ones((natoms, 3), dtype=bool)
                    for i in range(len(cnstr)):
                        idx = cnstr[i].a
                        masks[idx] = cnstr[i].mask
                    cnstr = masks
            else:
                fr_cols.remove('move_mask')
        arrays = {}
        for column in fr_cols:
            if column == 'positions':
                arrays[column] = pos
            elif column in atoms.arrays:
                arrays[column] = atoms.arrays[column]
            elif column == 'symbols':
                arrays[column] = np.array(symbols)
            elif column == 'move_mask':
                arrays[column] = cnstr
            else:
                raise ValueError('Missing array "%s"' % column)
        if write_results:
            for key in per_atom_results:
                if key not in fr_cols:
                    fr_cols += [key]
                else:
                    warnings.warn('write_xyz() overwriting array "{0}" present in atoms.arrays with stored results from calculator'.format(key))
            arrays.update(per_atom_results)
        comm, ncols, dtype, fmt = output_column_format(atoms, fr_cols, arrays, write_info, per_frame_results)
        if plain or comment != '':
            comm = comment.rstrip()
            if '\n' in comm:
                raise ValueError('Comment line should not have line breaks.')
        data = np.zeros(natoms, dtype)
        for column, ncol in zip(fr_cols, ncols):
            value = arrays[column]
            if ncol == 1:
                data[column] = np.squeeze(value)
            else:
                for c in range(ncol):
                    data[column + str(c)] = value[:, c]
        nat = natoms
        if vec_cell:
            nat -= nPBC
        fileobj.write('%d\n' % nat)
        fileobj.write('%s\n' % comm)
        for i in range(natoms):
            fileobj.write(fmt % tuple(data[i]))