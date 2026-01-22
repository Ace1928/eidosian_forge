import os
import subprocess
from warnings import warn
import numpy as np
from ase.calculators.calculator import (Calculator, FileIOCalculator,
from ase.io import write
from ase.io.vasp import write_vasp
from ase.parallel import world
from ase.units import Bohr, Hartree
class DFTD3(FileIOCalculator):
    """Grimme DFT-D3 calculator"""
    name = 'DFTD3'
    command = 'dftd3'
    dftd3_implemented_properties = ['energy', 'forces', 'stress']
    damping_methods = ['zero', 'bj', 'zerom', 'bjm']
    default_parameters = {'xc': None, 'grad': True, 'abc': False, 'cutoff': 95 * Bohr, 'cnthr': 40 * Bohr, 'old': False, 'damping': 'zero', 'tz': False, 's6': None, 'sr6': None, 's8': None, 'sr8': None, 'alpha6': None, 'a1': None, 'a2': None, 'beta': None}
    dftd3_flags = ('grad', 'pbc', 'abc', 'old', 'tz')

    def __init__(self, label='ase_dftd3', command=None, dft=None, atoms=None, comm=world, **kwargs):
        self.dft = None
        FileIOCalculator.__init__(self, restart=None, label=label, atoms=atoms, command=command, dft=dft, **kwargs)
        self.comm = comm

    def set(self, **kwargs):
        changed_parameters = {}
        if kwargs.get('func'):
            if kwargs.get('xc') and kwargs['func'] != kwargs['xc']:
                raise RuntimeError('Both "func" and "xc" were provided! Please provide at most one of these two keywords. The preferred keyword is "xc"; "func" is allowed for consistency with the CLI dftd3 interface.')
            if kwargs['func'] != self.parameters['xc']:
                changed_parameters['xc'] = kwargs['func']
            self.parameters['xc'] = kwargs['func']
        if 'dft' in kwargs:
            dft = kwargs.pop('dft')
            if dft is not self.dft:
                changed_parameters['dft'] = dft
            if dft is None:
                self.implemented_properties = self.dftd3_implemented_properties
            else:
                self.implemented_properties = dft.implemented_properties
            self.dft = dft
        if self.parameters['xc'] is None and self.dft is not None:
            if self.dft.parameters.get('xc'):
                self.parameters['xc'] = self.dft.parameters['xc']
        unknown_kwargs = set(kwargs) - set(self.default_parameters)
        if unknown_kwargs:
            warn('WARNING: Ignoring the following unknown keywords: {}'.format(', '.join(unknown_kwargs)))
        changed_parameters.update(FileIOCalculator.set(self, **kwargs))
        if self.parameters['damping'] is not None:
            self.parameters['damping'] = self.parameters['damping'].lower()
        if self.parameters['damping'] not in self.damping_methods:
            raise ValueError('Unknown damping method {}!'.format(self.parameters['damping']))
        elif self.parameters['old'] and self.parameters['damping'] != 'zero':
            raise ValueError('Only zero-damping can be used with the D2 dispersion correction method!')
        if self.parameters['cnthr'] > self.parameters['cutoff']:
            warn('WARNING: CN cutoff value of {cnthr} is larger than regular cutoff value of {cutoff}! Reducing CN cutoff to {cutoff}.'.format(cnthr=self.parameters['cnthr'], cutoff=self.parameters['cutoff']))
            self.parameters['cnthr'] = self.parameters['cutoff']
        if not self.parameters['grad']:
            for val in ['forces', 'stress']:
                if val in self.implemented_properties:
                    self.implemented_properties.remove(val)
        zero_damppars = {'s6', 'sr6', 's8', 'sr8', 'alpha6'}
        bj_damppars = {'s6', 'a1', 's8', 'a2', 'alpha6'}
        zerom_damppars = {'s6', 'sr6', 's8', 'beta', 'alpha6'}
        all_damppars = zero_damppars | bj_damppars | zerom_damppars
        self.custom_damp = False
        damping = self.parameters['damping']
        damppars = set(kwargs) & all_damppars
        if damppars:
            self.custom_damp = True
            if damping == 'zero':
                valid_damppars = zero_damppars
            elif damping in ['bj', 'bjm']:
                valid_damppars = bj_damppars
            elif damping == 'zerom':
                valid_damppars = zerom_damppars
            missing_damppars = valid_damppars - damppars
            if missing_damppars and missing_damppars != valid_damppars:
                raise ValueError('An incomplete set of custom damping parameters for the {} damping method was provided! Expected: {}; got: {}'.format(damping, ', '.join(valid_damppars), ', '.join(damppars)))
            if damppars - valid_damppars:
                warn('WARNING: The following damping parameters are not valid for the {} damping method and will be ignored: {}'.format(damping, ', '.join(damppars)))
        if self.parameters['xc'] and self.custom_damp:
            warn('WARNING: Custom damping parameters will be used instead of those parameterized for {}!'.format(self.parameters['xc']))
        if changed_parameters:
            self.results.clear()
        return changed_parameters

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        localparfile = os.path.join(self.directory, '.dftd3par.local')
        if world.rank == 0 and os.path.isfile(localparfile):
            os.remove(localparfile)
        self.write_input(self.atoms, properties, system_changes)
        command = self._generate_command()
        errorcode = 0
        if self.comm.rank == 0:
            with open(self.label + '.out', 'w') as fd:
                errorcode = subprocess.call(command, cwd=self.directory, stdout=fd)
        errorcode = self.comm.sum(errorcode)
        if errorcode:
            raise RuntimeError('%s returned an error: %d' % (self.name, errorcode))
        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties=properties, system_changes=system_changes)
        pbc = False
        if any(atoms.pbc):
            if not all(atoms.pbc):
                warn('WARNING! dftd3 can only calculate the dispersion energy of non-periodic or 3D-periodic systems. We will treat this system as 3D-periodic!')
            pbc = True
        if self.comm.rank == 0:
            if pbc:
                fname = os.path.join(self.directory, '{}.POSCAR'.format(self.label))
                write_vasp(fname, atoms, sort=True)
            else:
                fname = os.path.join(self.directory, '{}.xyz'.format(self.label))
                write(fname, atoms, format='xyz', parallel=False)
        if self.custom_damp:
            damppars = []
            damppars.append(str(float(self.parameters['s6'])))
            if self.parameters['damping'] in ['zero', 'zerom']:
                damppars.append(str(float(self.parameters['sr6'])))
            elif self.parameters['damping'] in ['bj', 'bjm']:
                damppars.append(str(float(self.parameters['a1'])))
            damppars.append(str(float(self.parameters['s8'])))
            if self.parameters['damping'] == 'zero':
                damppars.append(str(float(self.parameters['sr8'])))
            elif self.parameters['damping'] in ['bj', 'bjm']:
                damppars.append(str(float(self.parameters['a2'])))
            elif self.parameters['damping'] == 'zerom':
                damppars.append(str(float(self.parameters['beta'])))
            damppars.append(str(int(self.parameters['alpha6'])))
            if self.parameters['old']:
                damppars.append('2')
            elif self.parameters['damping'] == 'zero':
                damppars.append('3')
            elif self.parameters['damping'] == 'bj':
                damppars.append('4')
            elif self.parameters['damping'] == 'zerom':
                damppars.append('5')
            elif self.parameters['damping'] == 'bjm':
                damppars.append('6')
            damp_fname = os.path.join(self.directory, '.dftd3par.local')
            if self.comm.rank == 0:
                with open(damp_fname, 'w') as fd:
                    fd.write(' '.join(damppars))

    def read_results(self):
        outname = os.path.join(self.directory, self.label + '.out')
        energy = 0.0
        if self.comm.rank == 0:
            with open(outname, 'r') as fd:
                for line in fd:
                    if line.startswith(' program stopped'):
                        if 'functional name unknown' in line:
                            message = 'Unknown DFTD3 functional name "{}". Please check the dftd3.f source file for the list of known functionals and their spelling.'.format(self.parameters['xc'])
                        else:
                            message = 'dftd3 failed! Please check the {} output file and report any errors to the ASE developers.'.format(outname)
                        raise RuntimeError(message)
                    if line.startswith(' Edisp'):
                        parts = line.split()
                        assert parts[1][0] == '/'
                        index = 2 + parts[1][1:-1].split(',').index('au')
                        e_dftd3 = float(parts[index]) * Hartree
                        energy = e_dftd3
                        break
                else:
                    raise RuntimeError('Could not parse energy from dftd3 output, see file {}'.format(outname))
        self.results['energy'] = self.comm.sum(energy)
        self.results['free_energy'] = self.results['energy']
        if self.dft is not None:
            try:
                efree = self.dft.get_potential_energy(force_consistent=True)
                self.results['free_energy'] += efree
            except PropertyNotImplementedError:
                pass
        if self.parameters['grad']:
            forces = np.zeros((len(self.atoms), 3))
            forcename = os.path.join(self.directory, 'dftd3_gradient')
            if self.comm.rank == 0:
                with open(forcename, 'r') as fd:
                    for i, line in enumerate(fd):
                        forces[i] = np.array([float(x) for x in line.split()])
                forces *= -Hartree / Bohr
            self.comm.broadcast(forces, 0)
            if self.atoms.pbc.any():
                ind = np.argsort(self.atoms.get_chemical_symbols())
                forces[ind] = forces.copy()
            self.results['forces'] = forces
            if any(self.atoms.pbc):
                stress = np.zeros((3, 3))
                stressname = os.path.join(self.directory, 'dftd3_cellgradient')
                if self.comm.rank == 0:
                    with open(stressname, 'r') as fd:
                        for i, line in enumerate(fd):
                            for j, x in enumerate(line.split()):
                                stress[i, j] = float(x)
                    stress *= Hartree / Bohr / self.atoms.get_volume()
                    stress = np.dot(stress.T, self.atoms.cell)
                self.comm.broadcast(stress, 0)
                self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]

    def get_property(self, name, atoms=None, allow_calculation=True):
        dft_result = None
        if self.dft is not None:
            dft_result = self.dft.get_property(name, atoms, allow_calculation)
        dftd3_result = FileIOCalculator.get_property(self, name, atoms, allow_calculation)
        if dft_result is None and dftd3_result is None:
            return None
        elif dft_result is None:
            return dftd3_result
        elif dftd3_result is None:
            return dft_result
        else:
            return dft_result + dftd3_result

    def _generate_command(self):
        command = self.command.split()
        if any(self.atoms.pbc):
            command.append(self.label + '.POSCAR')
        else:
            command.append(self.label + '.xyz')
        if not self.custom_damp:
            xc = self.parameters.get('xc')
            if xc is None:
                xc = 'pbe'
            command += ['-func', xc.lower()]
        for arg in self.dftd3_flags:
            if self.parameters.get(arg):
                command.append('-' + arg)
        if any(self.atoms.pbc):
            command.append('-pbc')
        command += ['-cnthr', str(self.parameters['cnthr'] / Bohr)]
        command += ['-cutoff', str(self.parameters['cutoff'] / Bohr)]
        if not self.parameters['old']:
            command.append('-' + self.parameters['damping'])
        return command