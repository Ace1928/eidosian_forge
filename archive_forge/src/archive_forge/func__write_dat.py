from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _write_dat(self, force_write=True):
    """This export function write minimal information to
        a .dat file. If the atoms object is a trajectory, it will
        take the last image.
        """
    filename = self.label + '.dat'
    if self.atoms is None:
        raise Exception('No associated atoms object.')
    atoms = self.atoms
    parameters = self.parameters
    if isfile(filename) and (not force_write):
        raise Exception('Target input file already exists.')
    if 'xc' in parameters and 'xc_functional' in parameters and (parameters['xc'] != parameters['xc_functional']):
        raise Exception('Conflicting functionals defined! %s vs. %s' % (parameters['xc'], parameters['xc_functional']))
    fd = open(filename, 'w')
    fd.write('######################################################\n')
    fd.write('#ONETEP .dat file: %s\n' % filename)
    fd.write('#Created using the Atomic Simulation Environment (ASE)\n')
    fd.write('######################################################\n\n')
    fd.write('%BLOCK LATTICE_CART\n')
    fd.write('ang\n')
    for line in atoms.get_cell():
        fd.write('    %.10f %.10f %.10f\n' % tuple(line))
    fd.write('%ENDBLOCK LATTICE_CART\n\n\n')
    keyword = 'POSITIONS_ABS'
    positions = atoms.get_positions()
    tags = ['' if i == 0 else str(i) for i in atoms.get_tags()]
    pos_block = ['%s %8.6f %8.6f %8.6f' % (x + z, y[0], y[1], y[2]) for x, y, z in zip(atoms.get_chemical_symbols(), positions, tags)]
    fd.write('%%BLOCK %s\n' % keyword)
    fd.write('ang\n')
    for line in pos_block:
        fd.write('    %s\n' % line)
    fd.write('%%ENDBLOCK %s\n\n' % keyword)
    keyword = 'SPECIES'
    sp_block = ['%s %s %d %d %8.6f' % sp for sp in self.species]
    fd.write('%%BLOCK %s\n' % keyword)
    for line in sorted(sp_block):
        fd.write('    %s\n' % line)
    fd.write('%%ENDBLOCK %s\n\n' % keyword)
    if self.parameters['ngwf_radius_cond'] > 0 or len(self.species_cond) == len(self.species):
        keyword = 'SPECIES_COND'
        sp_block = ['%s %s %d %d %8.6f' % sp for sp in self.species_cond]
        fd.write('%%BLOCK %s\n' % keyword)
        for line in sorted(sp_block):
            fd.write('    %s\n' % line)
        fd.write('%%ENDBLOCK %s\n\n' % keyword)
    keyword = 'SPECIES_POT'
    fd.write('%%BLOCK %s\n' % keyword)
    for sp in sorted(self.pseudos):
        fd.write('    %s "%s"\n' % (sp[0], sp[1]))
    fd.write('%%ENDBLOCK %s\n\n' % keyword)
    keyword = 'SPECIES_ATOMIC_SET'
    fd.write('%%BLOCK %s\n' % keyword)
    for sp in sorted(self.solvers):
        fd.write('    %s "%s"\n' % (sp[0], sp[1]))
    fd.write('%%ENDBLOCK %s\n\n' % keyword)
    if self.parameters['ngwf_radius_cond'] > 0 or len(self.solvers_cond) == len(self.species):
        keyword = 'SPECIES_ATOMIC_SET_COND'
        fd.write('%%BLOCK %s\n' % keyword)
        for sp in sorted(self.solvers_cond):
            fd.write('    %s "%s"\n' % (sp[0], sp[1]))
        fd.write('%%ENDBLOCK %s\n\n' % keyword)
    if self.core_wfs:
        keyword = 'SPECIES_CORE_WF'
        fd.write('%%BLOCK %s\n' % keyword)
        for sp in sorted(self.core_wfs):
            fd.write('    %s "%s"\n' % (sp[0], sp[1]))
        fd.write('%%ENDBLOCK %s\n\n' % keyword)
    if 'bsunfld_calculate' in self.parameters:
        if 'species_bsunfld_groups' not in self.parameters:
            self.parameters['species_bsunfld_groups'] = self.atoms.get_chemical_symbols()
    for p, param in sorted(parameters.items()):
        if param is not None and p.lower() not in self._dummy_parameters:
            if p.lower() in self._block_parameters:
                keyword = p.upper()
                fd.write('\n%%BLOCK %s\n' % keyword)
                if p.lower() in self._path_parameters:
                    self.write_kpt_path(fd, param)
                elif p.lower() in self._group_parameters:
                    self.write_groups(fd, param)
                else:
                    fd.write('%s\n' % str(param))
                fd.write('%%ENDBLOCK %s\n\n' % keyword)
            else:
                fd.write('%s : %s\n' % (p, param))
        if p.upper() == 'XC':
            fd.write('xc_functional : %s\n' % param)
    fd.close()