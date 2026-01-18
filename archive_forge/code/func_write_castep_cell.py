import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def write_castep_cell(fd, atoms, positions_frac=False, force_write=False, precision=6, magnetic_moments=None, castep_cell=None):
    """
    This CASTEP export function write minimal information to
    a .cell file. If the atoms object is a trajectory, it will
    take the last image.

    Note that function has been altered in order to require a filedescriptor
    rather than a filename. This allows to use the more generic write()
    function from formats.py

    Note that the "force_write" keywords has no effect currently.

    Arguments:

        positions_frac: boolean. If true, positions are printed as fractional
                        rather than absolute. Default is false.
        castep_cell: if provided, overrides the existing CastepCell object in
                     the Atoms calculator
        precision: number of digits to which lattice and positions are printed
        magnetic_moments: if None, no SPIN values are initialised.
                          If 'initial', the values from
                          get_initial_magnetic_moments() are used.
                          If 'calculated', the values from
                          get_magnetic_moments() are used.
                          If an array of the same length as the atoms object,
                          its contents will be used as magnetic moments.
    """
    if atoms is None:
        warnings.warn('Atoms object not initialized')
        return False
    if isinstance(atoms, list):
        if len(atoms) > 1:
            atoms = atoms[-1]
    fd.write('#######################################################\n')
    fd.write('#CASTEP cell file: %s\n' % fd.name)
    fd.write('#Created using the Atomic Simulation Environment (ASE)#\n')
    fd.write('#######################################################\n\n')
    from ase.calculators.castep import Castep, CastepCell
    try:
        has_cell = isinstance(atoms.calc.cell, CastepCell)
    except AttributeError:
        has_cell = False
    if has_cell:
        cell = deepcopy(atoms.calc.cell)
    else:
        cell = Castep(keyword_tolerance=2).cell
    fformat = '%{0}.{1}f'.format(precision + 3, precision)
    cell_block_format = ' '.join([fformat] * 3)
    cell.lattice_cart = [cell_block_format % tuple(line) for line in atoms.get_cell()]
    if positions_frac:
        pos_keyword = 'positions_frac'
        positions = atoms.get_scaled_positions()
    else:
        pos_keyword = 'positions_abs'
        positions = atoms.get_positions()
    if atoms.has('castep_custom_species'):
        elems = atoms.get_array('castep_custom_species')
    else:
        elems = atoms.get_chemical_symbols()
    if atoms.has('castep_labels'):
        labels = atoms.get_array('castep_labels')
    else:
        labels = ['NULL'] * len(elems)
    if str(magnetic_moments).lower() == 'initial':
        magmoms = atoms.get_initial_magnetic_moments()
    elif str(magnetic_moments).lower() == 'calculated':
        magmoms = atoms.get_magnetic_moments()
    elif np.array(magnetic_moments).shape == (len(elems),):
        magmoms = np.array(magnetic_moments)
    else:
        magmoms = [0] * len(elems)
    pos_block = []
    pos_block_format = '%s ' + cell_block_format
    for i, el in enumerate(elems):
        xyz = positions[i]
        line = pos_block_format % tuple([el] + list(xyz))
        if magmoms[i] != 0:
            line += ' SPIN={0} '.format(magmoms[i])
        if labels[i].strip() not in ('NULL', ''):
            line += ' LABEL={0} '.format(labels[i])
        pos_block.append(line)
    setattr(cell, pos_keyword, pos_block)
    constraints = atoms.constraints
    if len(constraints):
        _supported_constraints = (FixAtoms, FixedPlane, FixedLine, FixCartesian)
        constr_block = []
        for constr in constraints:
            if not isinstance(constr, _supported_constraints):
                warnings.warn('Warning: you have constraints in your atoms, that are not supported by the CASTEP ase interface')
                break
            if isinstance(constr, FixAtoms):
                for i in constr.index:
                    try:
                        symbol = atoms.get_chemical_symbols()[i]
                        nis = atoms.calc._get_number_in_species(i)
                    except KeyError:
                        raise UserWarning('Unrecognized index in' + ' constraint %s' % constr)
                    for j in range(3):
                        L = '%6d %3s %3d   ' % (len(constr_block) + 1, symbol, nis)
                        L += ['1 0 0', '0 1 0', '0 0 1'][j]
                        constr_block += [L]
            elif isinstance(constr, FixCartesian):
                n = constr.a
                symbol = atoms.get_chemical_symbols()[n]
                nis = atoms.calc._get_number_in_species(n)
                for i, m in enumerate(constr.mask):
                    if m == 1:
                        continue
                    L = '%6d %3s %3d   ' % (len(constr_block) + 1, symbol, nis)
                    L += ' '.join(['1' if j == i else '0' for j in range(3)])
                    constr_block += [L]
            elif isinstance(constr, FixedPlane):
                n = constr.a
                symbol = atoms.get_chemical_symbols()[n]
                nis = atoms.calc._get_number_in_species(n)
                L = '%6d %3s %3d   ' % (len(constr_block) + 1, symbol, nis)
                L += ' '.join([str(d) for d in constr.dir])
                constr_block += [L]
            elif isinstance(constr, FixedLine):
                n = constr.a
                symbol = atoms.get_chemical_symbols()[n]
                nis = atoms.calc._get_number_in_species(n)
                direction = constr.dir
                (i1, v1), (i2, v2) = sorted(enumerate(direction), key=lambda x: abs(x[1]), reverse=True)[:2]
                n1 = np.zeros(3)
                n1[i2] = v1
                n1[i1] = -v2
                n1 = n1 / np.linalg.norm(n1)
                n2 = np.cross(direction, n1)
                l1 = '%6d %3s %3d   %f %f %f' % (len(constr_block) + 1, symbol, nis, n1[0], n1[1], n1[2])
                l2 = '%6d %3s %3d   %f %f %f' % (len(constr_block) + 2, symbol, nis, n2[0], n2[1], n2[2])
                constr_block += [l1, l2]
        cell.ionic_constraints = constr_block
    write_freeform(fd, cell)
    return True