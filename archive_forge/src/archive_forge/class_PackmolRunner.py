from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
@deprecated(PackmolBoxGen, 'PackmolRunner is being phased out in favor of the packmol I/O class.')
class PackmolRunner:
    """
    Wrapper for the Packmol software that can be used to pack various types of
    molecules into a one single unit.
    """

    def __init__(self, mols: list, param_list: list, input_file: str='pack.inp', tolerance: float=2.0, filetype: str='xyz', control_params: dict | None=None, auto_box: bool=True, output_file: str='packed.xyz', bin: str='packmol') -> None:
        """
        Args:
            mols:
                list of Molecules to pack
            input_file:
                name of the packmol input file
            tolerance:
                min distance between the atoms
            filetype:
                input/output structure file type
            control_params:
                packmol control parameters dictionary. Basically
                all parameters other than structure/atoms
            param_list:
                list of parameters containing dicts for each molecule
            auto_box:
                put the molecule assembly in a box
            output_file:
                output file name. The extension will be adjusted
                according to the filetype.
        """
        self.packmol_bin = bin.split()
        if not which(self.packmol_bin[-1]):
            raise RuntimeError("PackmolRunner requires the executable 'packmol' to be in the path. Please download packmol from https://github.com/leandromartinez98/packmol and follow the instructions in the README to compile. Don't forget to add the packmol binary to your path")
        self.mols = mols
        self.param_list = param_list
        self.input_file = input_file
        self.boxit = auto_box
        self.control_params = control_params or {'maxit': 20, 'nloop': 600}
        if not self.control_params.get('tolerance'):
            self.control_params['tolerance'] = tolerance
        if not self.control_params.get('filetype'):
            self.control_params['filetype'] = filetype
        if not self.control_params.get('output'):
            self.control_params['output'] = f'{output_file.split('.')[0]}.{self.control_params['filetype']}'
        if self.boxit:
            self._set_box()

    @staticmethod
    def _format_param_val(param_val) -> str:
        """
        Internal method to format values in the packmol parameter dictionaries.

        Args:
            param_val:
                Some object to turn into String

        Returns:
            String representation of the object
        """
        if isinstance(param_val, list):
            return ' '.join((str(x) for x in param_val))
        return str(param_val)

    def _set_box(self) -> None:
        """Set the box size for the molecular assembly."""
        net_volume = 0.0
        for idx, mol in enumerate(self.mols):
            length = max((np.max(mol.cart_coords[:, i]) - np.min(mol.cart_coords[:, i]) for i in range(3))) + 2.0
            net_volume += length ** 3.0 * float(self.param_list[idx]['number'])
        length = net_volume ** (1 / 3)
        for idx, _mol in enumerate(self.mols):
            self.param_list[idx]['inside box'] = f'0.0 0.0 0.0 {length} {length} {length}'

    def _write_input(self, input_dir: str='.') -> None:
        """
        Write the packmol input file to the input directory.

        Args:
            input_dir (str): path to the input directory
        """
        with open(f'{input_dir}/{self.input_file}', mode='w', encoding='utf-8') as inp:
            for key, val in self.control_params.items():
                inp.write(f'{key} {self._format_param_val(val)}\n')
            for idx, mol in enumerate(self.mols):
                filename = os.path.join(input_dir, f'{idx}.{self.control_params['filetype']}')
                if self.control_params['filetype'] == 'pdb':
                    self.write_pdb(mol, filename, num=idx + 1)
                else:
                    a = BabelMolAdaptor(mol)
                    pm = pybel.Molecule(a.openbabel_mol)
                    pm.write(self.control_params['filetype'], filename=filename, overwrite=True)
                inp.write('\n')
                inp.write(f'structure {os.path.join(input_dir, str(idx))}.{self.control_params['filetype']}\n')
                for key, val in self.param_list[idx].items():
                    inp.write(f'  {key} {self._format_param_val(val)}\n')
                inp.write('end structure\n')

    def run(self, site_property: str | None=None) -> Molecule:
        """
        Write the input file to the scratch directory, run packmol and return
        the packed molecule to the current working directory.

        Args:
            site_property (str): if set then the specified site property
                for the final packed molecule will be restored.

        Returns:
            Molecule object
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            self._write_input(input_dir=scratch_dir)
            with open(os.path.join(scratch_dir, self.input_file)) as packmol_input, Popen(self.packmol_bin, stdin=packmol_input, stdout=PIPE, stderr=PIPE) as proc:
                stdout, stderr = proc.communicate()
            output_file = self.control_params['output']
            if os.path.isfile(output_file):
                packed_mol = BabelMolAdaptor.from_file(output_file, self.control_params['filetype'])
                packed_mol = packed_mol.pymatgen_mol
                print(f'packed molecule written to {self.control_params['output']}')
                if site_property:
                    packed_mol = self.restore_site_properties(site_property=site_property, filename=output_file)
                return packed_mol
            raise RuntimeError(f'Packmol execution failed. {stdout.decode}\n{stderr.decode}')

    @staticmethod
    def write_pdb(mol: Molecule, filename: str, name: str | None=None, num=None) -> None:
        """Dump the molecule into pdb file with custom residue name and number."""
        with ScratchDir('.'):
            mol.to(fmt='pdb', filename='tmp.pdb')
            bma = BabelMolAdaptor.from_file('tmp.pdb', 'pdb')
        num = num or 1
        name = name or f'ml{num}'
        pbm = pybel.Molecule(bma._ob_mol)
        for x in pbm.residues:
            x.OBResidue.SetName(name)
            x.OBResidue.SetNum(num)
        pbm.write(format='pdb', filename=filename, overwrite=True)

    def _set_residue_map(self) -> None:
        """Map each residue to the corresponding molecule."""
        self.map_residue_to_mol = {}
        lookup = {}
        for idx, mol in enumerate(self.mols):
            if mol.formula not in lookup:
                mol.translate_sites(indices=range(len(mol)), vector=-mol.center_of_mass)
                lookup[mol.formula] = mol.copy()
            self.map_residue_to_mol[f'ml{idx + 1}'] = lookup[mol.formula]

    def convert_obatoms_to_molecule(self, atoms: Sequence, residue_name: str | None=None, site_property: str='ff_map') -> Molecule:
        """
        Convert list of openbabel atoms to Molecule.

        Args:
            atoms ([OBAtom]): list of OBAtom objects
            residue_name (str): the key in self.map_residue_to_mol. Used to
                restore the site properties in the final packed molecule.
            site_property (str): the site property to be restored.

        Returns:
            Molecule object
        """
        if residue_name is not None and (not hasattr(self, 'map_residue_to_mol')):
            self._set_residue_map()
        coords = []
        zs = []
        for atm in atoms:
            coords.append(list(atm.coords))
            zs.append(atm.atomicnum)
        mol = Molecule(zs, coords)
        if residue_name is not None:
            props = []
            ref = self.map_residue_to_mol[residue_name].copy()
            assert len(mol) == len(ref)
            assert ref.formula == mol.formula
            for idx, site in enumerate(mol):
                assert site.specie.symbol == ref[idx].specie.symbol
                props.append(getattr(ref[idx], site_property))
            mol.add_site_property(site_property, props)
        return mol

    def restore_site_properties(self, site_property: str='ff_map', filename: str | None=None) -> Molecule:
        """
        Restore the site properties for the final packed molecule.

        Args:
            site_property (str):
            filename (str): path to the final packed molecule.

        Returns:
            Molecule
        """
        if not self.control_params['filetype'] == 'pdb':
            raise ValueError('site properties can only be restored for pdb files.')
        filename = filename or self.control_params['output']
        bma = BabelMolAdaptor.from_file(filename, 'pdb')
        pbm = pybel.Molecule(bma._ob_mol)
        assert len(pbm.residues) == sum((x['number'] for x in self.param_list))
        packed_mol = self.convert_obatoms_to_molecule(pbm.residues[0].atoms, residue_name=pbm.residues[0].name, site_property=site_property)
        for resid in pbm.residues[1:]:
            mol = self.convert_obatoms_to_molecule(resid.atoms, residue_name=resid.name, site_property=site_property)
            for site in mol:
                packed_mol.append(site.species, site.coords, properties=site.properties)
        return packed_mol