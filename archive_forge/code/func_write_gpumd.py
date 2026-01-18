import numpy as np
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase import Atoms
def write_gpumd(fd, atoms, maximum_neighbors=None, cutoff=None, groupings=None, use_triclinic=False):
    """
    Writes atoms into GPUMD input format.

    Parameters
    ----------
    fd : file
        File like object to which the atoms object should be written
    atoms : Atoms
        Input structure
    maximum_neighbors: int
        Maximum number of neighbors any atom can ever have (not relevant when
        using force constant potentials)
    cutoff: float
        Initial\xa0cutoff distance used for building the neighbor list (not
        relevant when using force constant potentials)
    groupings : list[list[list[int]]]
        Groups into which the individual atoms should be divided in the form of
        a list of list of lists. Specifically, the outer list corresponds to
        the grouping methods, of which there can be three at the most, which
        contains a list of groups in the form of lists of site indices. The
        sum of the lengths of the latter must be the same as the total number
        of atoms.
    use_triclinic: bool
        Use format for triclinic cells

    Raises
    ------
    ValueError
        Raised if parameters are incompatible
    """
    if atoms.get_velocities() is None:
        has_velocity = 0
    else:
        has_velocity = 1
        velocities = atoms.get_velocities()
    if groupings is None:
        number_of_grouping_methods = 0
    else:
        number_of_grouping_methods = len(groupings)
        if number_of_grouping_methods > 3:
            raise ValueError('There can be no more than 3 grouping methods!')
        for g, grouping in enumerate(groupings):
            all_indices = [i for group in grouping for i in group]
            if len(all_indices) != len(atoms) or set(all_indices) != set(range(len(atoms))):
                raise ValueError('The indices listed in grouping method {} are not compatible with the input structure!'.format(g))
    if maximum_neighbors is None:
        if cutoff is None:
            cutoff = 0.1
            maximum_neighbors = 1
        else:
            nl = NeighborList([cutoff / 2] * len(atoms), skin=2, bothways=True)
            nl.update(atoms)
            maximum_neighbors = 0
            for atom in atoms:
                maximum_neighbors = max(maximum_neighbors, len(nl.get_neighbors(atom.index)[0]))
                maximum_neighbors *= 2
    lines = []
    if atoms.cell.orthorhombic and (not use_triclinic):
        triclinic = 0
    else:
        triclinic = 1
    lines.append('{} {} {} {} {} {}'.format(len(atoms), maximum_neighbors, cutoff, triclinic, has_velocity, number_of_grouping_methods))
    if triclinic:
        lines.append((' {}' * 12)[1:].format(*atoms.pbc.astype(int), *atoms.cell[:].flatten()))
    else:
        lines.append((' {}' * 6)[1:].format(*atoms.pbc.astype(int), *atoms.cell.lengths()))
    symbol_type_map = {}
    for symbol in atoms.get_chemical_symbols():
        if symbol not in symbol_type_map:
            symbol_type_map[symbol] = len(symbol_type_map)
    for a, atm in enumerate(atoms):
        t = symbol_type_map[atm.symbol]
        line = (' {}' * 5)[1:].format(t, *atm.position, atm.mass)
        if has_velocity:
            line += (' {}' * 3).format(*velocities[a])
        if groupings is not None:
            for grouping in groupings:
                for i, group in enumerate(grouping):
                    if a in group:
                        line += ' {}'.format(i)
                        break
        lines.append(line)
    fd.write('\n'.join(lines))