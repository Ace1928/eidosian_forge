from __future__ import annotations
import warnings
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.core.structure import Molecule, Structure
class AseAtomsAdaptor:
    """Adaptor serves as a bridge between ASE Atoms and pymatgen objects."""

    @staticmethod
    def get_atoms(structure: SiteCollection, msonable: bool=True, **kwargs) -> MSONAtoms | Atoms:
        """
        Returns ASE Atoms object from pymatgen structure or molecule.

        Args:
            structure (SiteCollection): pymatgen Structure or Molecule
            msonable (bool): Whether to return an MSONAtoms object, which is MSONable.
            **kwargs: passed to the ASE Atoms constructor

        Returns:
            Atoms: ASE Atoms object
        """
        if no_ase_err:
            raise no_ase_err
        if not structure.is_ordered:
            raise ValueError('ASE Atoms only supports ordered structures')
        symbols = [str(site.specie.symbol) for site in structure]
        positions = [site.coords for site in structure]
        if hasattr(structure, 'lattice'):
            pbc = True
            cell = structure.lattice.matrix
        else:
            pbc = False
            cell = None
        atoms = Atoms(symbols=symbols, positions=positions, pbc=pbc, cell=cell, **kwargs)
        if msonable:
            atoms = MSONAtoms(atoms)
        if 'tags' in structure.site_properties:
            atoms.set_tags(structure.site_properties['tags'])
        if 'charge' in structure.site_properties:
            initial_charges = structure.site_properties['charge']
            atoms.set_initial_charges(initial_charges)
        if 'magmom' in structure.site_properties:
            initial_magmoms = structure.site_properties['magmom']
            atoms.set_initial_magnetic_moments(initial_magmoms)
        if isinstance(structure, Molecule):
            atoms.charge = structure.charge
            atoms.spin_multiplicity = structure.spin_multiplicity
        oxi_states: list[float | None] = [getattr(site.specie, 'oxi_state', None) for site in structure]
        if 'selective_dynamics' in structure.site_properties:
            fix_atoms = []
            for site in structure:
                selective_dynamics: ArrayLike = site.properties.get('selective_dynamics')
                if isinstance(selective_dynamics, Iterable) and True in selective_dynamics and (False in selective_dynamics):
                    raise ValueError(f'ASE FixAtoms constraint does not support selective dynamics in only some dimensions. Remove the selective_dynamics={selective_dynamics!r} and try again if you do not need them.')
                is_fixed = bool(~np.all(site.properties['selective_dynamics']))
                fix_atoms.append(is_fixed)
        else:
            fix_atoms = None
        if fix_atoms is not None:
            atoms.set_constraint(FixAtoms(mask=fix_atoms))
        for prop in structure.site_properties:
            if prop not in ['magmom', 'charge', 'final_magmom', 'final_charge', 'selective_dynamics']:
                atoms.set_array(prop, np.array(structure.site_properties[prop]))
        if any(oxi_states):
            atoms.set_array('oxi_states', np.array(oxi_states))
        if (properties := getattr(structure, 'properties')):
            atoms.info = properties
        if isinstance(atoms.info.get('spacegroup'), dict):
            atoms.info['spacegroup'] = Spacegroup(atoms.info['spacegroup']['number'], setting=atoms.info['spacegroup'].get('setting', 1))
        if (calc := getattr(structure, 'calc', None)):
            atoms.calc = calc
        else:
            charges = structure.site_properties.get('final_charge')
            magmoms = structure.site_properties.get('final_magmom')
            if charges or magmoms:
                calc = SinglePointDFTCalculator(atoms, magmoms=magmoms, charges=charges)
                atoms.calc = calc
        return atoms

    @staticmethod
    def get_structure(atoms: Atoms, cls: type[Structure]=Structure, **cls_kwargs) -> Structure:
        """
        Returns pymatgen structure from ASE Atoms.

        Args:
            atoms: ASE Atoms object
            cls: The Structure class to instantiate (defaults to pymatgen Structure)
            **cls_kwargs: Any additional kwargs to pass to the cls

        Returns:
            Structure: Equivalent pymatgen Structure
        """
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        lattice = atoms.get_cell()
        tags = atoms.get_tags() if atoms.has('tags') else None
        if getattr(atoms, 'calc', None) is not None and getattr(atoms.calc, 'results', None) is not None:
            charges = atoms.calc.results.get('charges')
            magmoms = atoms.calc.results.get('magmoms')
        else:
            magmoms = charges = None
        initial_charges = atoms.get_initial_charges() if atoms.has('initial_charges') else None
        initial_magmoms = atoms.get_initial_magnetic_moments() if atoms.has('initial_magmoms') else None
        oxi_states = atoms.get_array('oxi_states') if atoms.has('oxi_states') else None
        if atoms.constraints:
            unsupported_constraint_type = False
            constraint_indices = []
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    constraint_indices.extend(constraint.get_indices().tolist())
                else:
                    unsupported_constraint_type = True
            if unsupported_constraint_type:
                warnings.warn('Only FixAtoms is supported by Pymatgen. Other constraints will not be set.', UserWarning)
            sel_dyn = [[False] * 3 if atom.index in constraint_indices else [True] * 3 for atom in atoms]
        else:
            sel_dyn = None
        if atoms.info.get('spacegroup') and isinstance(atoms.info['spacegroup'], Spacegroup):
            atoms.info['spacegroup'] = atoms.info['spacegroup'].todict()
        properties = getattr(atoms, 'info', {})
        if cls == Molecule:
            structure = cls(symbols, positions, properties=properties, **cls_kwargs)
        else:
            structure = cls(lattice, symbols, positions, coords_are_cartesian=True, properties=properties, **cls_kwargs)
        if (calc := getattr(atoms, 'calc', None)):
            structure.calc = calc
        if initial_charges is not None:
            structure.add_site_property('charge', initial_charges)
        if charges is not None:
            structure.add_site_property('final_charge', charges)
        if magmoms is not None:
            structure.add_site_property('final_magmom', magmoms)
        if initial_magmoms is not None:
            structure.add_site_property('magmom', initial_magmoms)
        if sel_dyn is not None and ~np.all(sel_dyn):
            structure.add_site_property('selective_dynamics', sel_dyn)
        if tags is not None:
            structure.add_site_property('tags', tags)
        if oxi_states is not None:
            structure.add_oxidation_state_by_site(oxi_states)
        for prop in atoms.arrays:
            if prop not in ['numbers', 'positions', 'magmom', 'initial_charges', 'initial_magmoms', 'final_magmom', 'charge', 'final_charge', 'oxi_states']:
                structure.add_site_property(prop, atoms.get_array(prop).tolist())
        return structure

    @staticmethod
    def get_molecule(atoms: Atoms, cls: type[Molecule]=Molecule, **cls_kwargs) -> Molecule:
        """
        Returns pymatgen molecule from ASE Atoms.

        Args:
            atoms: ASE Atoms object
            cls: The Molecule class to instantiate (defaults to pymatgen molecule)
            **cls_kwargs: Any additional kwargs to pass to the cls

        Returns:
            Molecule: Equivalent pymatgen.core.structure.Molecule
        """
        molecule = AseAtomsAdaptor.get_structure(atoms, cls=cls, **cls_kwargs)
        try:
            charge = atoms.charge
        except AttributeError:
            charge = round(np.sum(atoms.get_initial_charges())) if atoms.has('initial_charges') else 0
        try:
            spin_mult = atoms.spin_multiplicity
        except AttributeError:
            spin_mult = round(np.sum(atoms.get_initial_magnetic_moments())) + 1 if atoms.has('initial_magmoms') else 1
        molecule.set_charge_and_spin(charge, spin_multiplicity=spin_mult)
        return molecule