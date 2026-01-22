import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def _write_species(self, fd, atoms):
    """Write input related the different species.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
    species, species_numbers = self.species(atoms)
    if self['pseudo_path'] is not None:
        pseudo_path = self['pseudo_path']
    elif 'SIESTA_PP_PATH' in os.environ:
        pseudo_path = os.environ['SIESTA_PP_PATH']
    else:
        mess = "Please set the environment variable 'SIESTA_PP_PATH'"
        raise Exception(mess)
    fd.write(format_fdf('NumberOfSpecies', len(species)))
    fd.write(format_fdf('NumberOfAtoms', len(atoms)))
    pao_basis = []
    chemical_labels = []
    basis_sizes = []
    synth_blocks = []
    for species_number, spec in enumerate(species):
        species_number += 1
        symbol = spec['symbol']
        atomic_number = atomic_numbers[symbol]
        if spec['pseudopotential'] is None:
            if self.pseudo_qualifier() == '':
                label = symbol
                pseudopotential = label + '.psf'
            else:
                label = '.'.join([symbol, self.pseudo_qualifier()])
                pseudopotential = label + '.psf'
        else:
            pseudopotential = spec['pseudopotential']
            label = os.path.basename(pseudopotential)
            label = '.'.join(label.split('.')[:-1])
        if not os.path.isabs(pseudopotential):
            pseudopotential = join(pseudo_path, pseudopotential)
        if not os.path.exists(pseudopotential):
            mess = "Pseudopotential '%s' not found" % pseudopotential
            raise RuntimeError(mess)
        name = os.path.basename(pseudopotential)
        name = name.split('.')
        name.insert(-1, str(species_number))
        if spec['ghost']:
            name.insert(-1, 'ghost')
            atomic_number = -atomic_number
        name = '.'.join(name)
        pseudo_targetpath = self.getpath(name)
        if join(os.getcwd(), name) != pseudopotential:
            if islink(pseudo_targetpath) or isfile(pseudo_targetpath):
                os.remove(pseudo_targetpath)
            symlink_pseudos = self['symlink_pseudos']
            if symlink_pseudos is None:
                symlink_pseudos = not os.name == 'nt'
            if symlink_pseudos:
                os.symlink(pseudopotential, pseudo_targetpath)
            else:
                shutil.copy(pseudopotential, pseudo_targetpath)
        if not spec['excess_charge'] is None:
            atomic_number += 200
            n_atoms = sum(np.array(species_numbers) == species_number)
            paec = float(spec['excess_charge']) / n_atoms
            vc = get_valence_charge(pseudopotential)
            fraction = float(vc + paec) / vc
            pseudo_head = name[:-4]
            fractional_command = os.environ['SIESTA_UTIL_FRACTIONAL']
            cmd = '%s %s %.7f' % (fractional_command, pseudo_head, fraction)
            os.system(cmd)
            pseudo_head += '-Fraction-%.5f' % fraction
            synth_pseudo = pseudo_head + '.psf'
            synth_block_filename = pseudo_head + '.synth'
            os.remove(name)
            shutil.copyfile(synth_pseudo, name)
            synth_block = read_vca_synth_block(synth_block_filename, species_number=species_number)
            synth_blocks.append(synth_block)
        if len(synth_blocks) > 0:
            fd.write(format_fdf('SyntheticAtoms', list(synth_blocks)))
        label = '.'.join(np.array(name.split('.'))[:-1])
        string = '    %d %d %s' % (species_number, atomic_number, label)
        chemical_labels.append(string)
        if isinstance(spec['basis_set'], PAOBasisBlock):
            pao_basis.append(spec['basis_set'].script(label))
        else:
            basis_sizes.append(('    ' + label, spec['basis_set']))
    fd.write(format_fdf('ChemicalSpecieslabel', chemical_labels))
    fd.write('\n')
    fd.write(format_fdf('PAO.Basis', pao_basis))
    fd.write(format_fdf('PAO.BasisSizes', basis_sizes))
    fd.write('\n')