from io import StringIO
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import InputError, ReadError
from ase.calculators.calculator import CalculatorSetupError
import multiprocessing
from ase import io
import numpy as np
import json
from ase.units import Bohr, Hartree
import warnings
import os
class Psi4(Calculator):
    """
    An ase calculator for the popular open source Q-chem code
    psi4.
    method is the generic input for whatever method you wish to use, thus
    and quantum chemistry method implemented in psi4 can be input
    (i.e. ccsd(t))

    also note that you can always use the in-built psi4 module through:
    calc.psi4
    """
    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True
    default_parameters = {'basis': 'aug-cc-pvtz', 'method': 'hf', 'symmetry': 'c1'}

    def __init__(self, restart=None, ignore_bad_restart=False, label='psi4-calc', atoms=None, command=None, **kwargs):
        Calculator.__init__(self, restart=restart, ignore_bad_restart=ignore_bad_restart, label=label, atoms=atoms, command=command, **kwargs)
        import psi4
        self.psi4 = psi4
        self.set_psi4(atoms=atoms)

    def set_psi4(self, atoms=None):
        """
        This function sets the imported psi4 module to the settings the user
        defines
        """
        if 'PSI_SCRATCH' in os.environ:
            pass
        elif self.parameters.get('PSI_SCRATCH'):
            os.environ['PSI_SCRATCH'] = self.parameters['PSI_SCRATCH']
        if self.parameters.get('reference') is not None:
            self.psi4.set_options({'reference': self.parameters['reference']})
        if self.parameters.get('memory') is not None:
            self.psi4.set_memory(self.parameters['memory'])
        nthreads = self.parameters.get('num_threads', 1)
        if nthreads == 'max':
            nthreads = multiprocessing.cpu_count()
        self.psi4.set_num_threads(nthreads)
        if 'kpts' in self.parameters:
            raise InputError('psi4 is a non-periodic code, and thus does not require k-points. Please remove this argument.')
        if self.parameters['method'] == 'LDA':
            self.parameters['method'] = 'svwn'
        if 'nbands' in self.parameters:
            raise InputError('psi4 does not support the keyword "nbands"')
        if 'smearing' in self.parameters:
            raise InputError('Finite temperature DFT is not implemented in psi4 currently, thus a smearing argument cannot be utilized. please remove this argument')
        if 'xc' in self.parameters:
            raise InputError('psi4 does not accept the `xc` argument please use the `method` argument instead')
        if atoms is None:
            if self.atoms is None:
                return None
            else:
                atoms = self.atoms
        if self.atoms is None:
            self.atoms = atoms
        geomline = '{}\t{:.15f}\t{:.15f}\t{:.15f}'
        geom = [geomline.format(atom.symbol, *atom.position) for atom in atoms]
        geom.append('symmetry {}'.format(self.parameters['symmetry']))
        geom.append('units angstrom')
        charge = self.parameters.get('charge')
        mult = self.parameters.get('multiplicity')
        if mult is None:
            mult = 1
            if charge is not None:
                warnings.warn('A charge was provided without a spin multiplicity. A multiplicity of 1 is assumed')
        if charge is None:
            charge = 0
        geom.append('{} {}'.format(charge, mult))
        geom.append('no_reorient')
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        self.molecule = self.psi4.geometry('\n'.join(geom))

    def read(self, label):
        """Read psi4 outputs made from this ASE calculator
        """
        filename = label + '.dat'
        if not os.path.isfile(filename):
            raise ReadError('Could not find the psi4 output file: ' + filename)
        with open(filename, 'r') as fd:
            txt = fd.read()
        if '!ASE Information\n' not in txt:
            raise Exception('The output file {} could not be read because the file does not contain the "!ASE Information" lines inserted by this calculator. This likely means the output file was not made using this ASE calculator or has since been modified and thus cannot be read.'.format(filename))
        info = txt.split('!ASE Information\n')[1]
        info = info.split('!')[0]
        saved_dict = json.loads(info)
        with StringIO(str(saved_dict['atoms'])) as g:
            self.atoms = io.read(g, format='json')
        self.parameters = saved_dict['parameters']
        self.results = saved_dict['results']
        if 'forces' in self.results:
            self.results['forces'] = np.array(self.results['forces'])

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes, symmetry='c1'):
        Calculator.calculate(self, atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to perform a calculation')
        atoms = self.atoms
        if atoms.get_initial_magnetic_moments().any():
            self.parameters['reference'] = 'uhf'
            self.parameters['multiplicity'] = None
        self.set_psi4(atoms=atoms)
        self.psi4.core.set_output_file(self.label + '.dat', False)
        method = self.parameters['method']
        basis = self.parameters['basis']
        if 'forces' in properties:
            grad, wf = self.psi4.driver.gradient('{}/{}'.format(method, basis), return_wfn=True)
            energy = wf.energy()
            self.results['energy'] = energy * Hartree
            self.results['forces'] = -1 * np.array(grad) * Hartree / Bohr
        elif 'energy' in properties:
            energy = self.psi4.energy('{}/{}'.format(method, basis), molecule=self.molecule)
            self.results['energy'] = energy * Hartree
        save_atoms = self.atoms.copy()
        with StringIO() as fd:
            io.write(fd, save_atoms, format='json')
            json_atoms = fd.getvalue()
        save_results = self.results.copy()
        if 'forces' in save_results:
            save_results['forces'] = save_results['forces'].tolist()
        save_dict = {'parameters': self.parameters, 'results': save_results, 'atoms': json_atoms}
        self.psi4.core.print_out('!ASE Information\n')
        self.psi4.core.print_out(json.dumps(save_dict))
        self.psi4.core.print_out('!')