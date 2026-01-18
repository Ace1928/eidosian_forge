from ase.units import Bohr
def write_turbomole(fd, atoms):
    """ Method to write turbomole coord file
    """
    from ase.constraints import FixAtoms
    coord = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    symbols = [element if element != 'X' else 'Q' for element in symbols]
    fix_indices = set()
    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_indices.update(constr.get_indices())
    fix_str = []
    for i in range(len(atoms)):
        if i in fix_indices:
            fix_str.append('f')
        else:
            fix_str.append('')
    fd.write('$coord\n')
    for (x, y, z), s, fix in zip(coord, symbols, fix_str):
        fd.write('%20.14f  %20.14f  %20.14f      %2s  %2s \n' % (x / Bohr, y / Bohr, z / Bohr, s.lower(), fix))
    fd.write('$end\n')