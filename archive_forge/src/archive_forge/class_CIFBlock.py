import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
class CIFBlock(collections.abc.Mapping):
    """A block (i.e., a single system) in a crystallographic information file.

    Use this object to query CIF tags or import information as ASE objects."""
    cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']

    def __init__(self, name: str, tags: Dict[str, CIFData]):
        self.name = name
        self._tags = tags

    def __repr__(self) -> str:
        tags = set(self._tags)
        return f'CIFBlock({self.name}, tags={tags})'

    def __getitem__(self, key: str) -> CIFData:
        return self._tags[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tags)

    def __len__(self) -> int:
        return len(self._tags)

    def get(self, key, default=None):
        return self._tags.get(key, default)

    def get_cellpar(self) -> Optional[List]:
        try:
            return [self[tag] for tag in self.cell_tags]
        except KeyError:
            return None

    def get_cell(self) -> Cell:
        cellpar = self.get_cellpar()
        if cellpar is None:
            return Cell.new([0, 0, 0])
        return Cell.new(cellpar)

    def _raw_scaled_positions(self) -> Optional[np.ndarray]:
        coords = [self.get(name) for name in ['_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']]
        if None in coords:
            return None
        return np.array(coords).T

    def _raw_positions(self) -> Optional[np.ndarray]:
        coords = [self.get('_atom_site_cartn_x'), self.get('_atom_site_cartn_y'), self.get('_atom_site_cartn_z')]
        if None in coords:
            return None
        return np.array(coords).T

    def _get_site_coordinates(self):
        scaled = self._raw_scaled_positions()
        if scaled is not None:
            return ('scaled', scaled)
        cartesian = self._raw_positions()
        if cartesian is None:
            raise NoStructureData('No positions found in structure')
        return ('cartesian', cartesian)

    def _get_symbols_with_deuterium(self):
        labels = self._get_any(['_atom_site_type_symbol', '_atom_site_label'])
        if labels is None:
            raise NoStructureData('No symbols')
        symbols = []
        for label in labels:
            if label == '.' or label == '?':
                raise NoStructureData('Symbols are undetermined')
            match = re.search('([A-Z][a-z]?)', label)
            symbol = match.group(0)
            symbols.append(symbol)
        return symbols

    def get_symbols(self) -> List[str]:
        symbols = self._get_symbols_with_deuterium()
        return [symbol if symbol != 'D' else 'H' for symbol in symbols]

    def _where_deuterium(self):
        return np.array([symbol == 'D' for symbol in self._get_symbols_with_deuterium()], bool)

    def _get_masses(self) -> Optional[np.ndarray]:
        mask = self._where_deuterium()
        if not any(mask):
            return None
        symbols = self.get_symbols()
        masses = Atoms(symbols).get_masses()
        masses[mask] = 2.01355
        return masses

    def _get_any(self, names):
        for name in names:
            if name in self:
                return self[name]
        return None

    def _get_spacegroup_number(self):
        return self._get_any(['_space_group.it_number', '_space_group_it_number', '_symmetry_int_tables_number'])

    def _get_spacegroup_name(self):
        hm_symbol = self._get_any(['_space_group_name_h-m_alt', '_symmetry_space_group_name_h-m', '_space_group.Patterson_name_h-m', '_space_group.patterson_name_h-m'])
        hm_symbol = old_spacegroup_names.get(hm_symbol, hm_symbol)
        return hm_symbol

    def _get_sitesym(self):
        sitesym = self._get_any(['_space_group_symop_operation_xyz', '_space_group_symop.operation_xyz', '_symmetry_equiv_pos_as_xyz'])
        if isinstance(sitesym, str):
            sitesym = [sitesym]
        return sitesym

    def _get_fractional_occupancies(self):
        return self.get('_atom_site_occupancy')

    def _get_setting(self) -> Optional[int]:
        setting_str = self.get('_symmetry_space_group_setting')
        if setting_str is None:
            return None
        setting = int(setting_str)
        if setting not in [1, 2]:
            raise ValueError(f'Spacegroup setting must be 1 or 2, not {setting}')
        return setting

    def get_spacegroup(self, subtrans_included) -> Spacegroup:
        no = self._get_spacegroup_number()
        hm_symbol = self._get_spacegroup_name()
        sitesym = self._get_sitesym()
        setting = 1
        spacegroup = 1
        if sitesym is not None:
            subtrans = [(0.0, 0.0, 0.0)] if subtrans_included else None
            spacegroup = spacegroup_from_data(no=no, symbol=hm_symbol, sitesym=sitesym, subtrans=subtrans, setting=setting)
        elif no is not None:
            spacegroup = no
        elif hm_symbol is not None:
            spacegroup = hm_symbol
        else:
            spacegroup = 1
        setting_std = self._get_setting()
        setting_name = None
        if '_symmetry_space_group_setting' in self:
            assert setting_std is not None
            setting = setting_std
        elif '_space_group_crystal_system' in self:
            setting_name = self['_space_group_crystal_system']
        elif '_symmetry_cell_setting' in self:
            setting_name = self['_symmetry_cell_setting']
        if setting_name:
            no = Spacegroup(spacegroup).no
            if no in rhombohedral_spacegroups:
                if setting_name == 'hexagonal':
                    setting = 1
                elif setting_name in ('trigonal', 'rhombohedral'):
                    setting = 2
                else:
                    warnings.warn('unexpected crystal system %r for space group %r' % (setting_name, spacegroup))
            else:
                warnings.warn('crystal system %r is not interpreted for space group %r. This may result in wrong setting!' % (setting_name, spacegroup))
        spg = Spacegroup(spacegroup, setting)
        if no is not None:
            assert int(spg) == no, (int(spg), no)
        return spg

    def get_unsymmetrized_structure(self) -> Atoms:
        """Return Atoms without symmetrizing coordinates.

        This returns a (normally) unphysical Atoms object
        corresponding only to those coordinates included
        in the CIF file, useful for e.g. debugging.

        This method may change behaviour in the future."""
        symbols = self.get_symbols()
        coordtype, coords = self._get_site_coordinates()
        atoms = Atoms(symbols=symbols, cell=self.get_cell(), masses=self._get_masses())
        if coordtype == 'scaled':
            atoms.set_scaled_positions(coords)
        else:
            assert coordtype == 'cartesian'
            atoms.positions[:] = coords
        return atoms

    def has_structure(self):
        """Whether this CIF block has an atomic configuration."""
        try:
            self.get_symbols()
            self._get_site_coordinates()
        except NoStructureData:
            return False
        else:
            return True

    def get_atoms(self, store_tags=False, primitive_cell=False, subtrans_included=True, fractional_occupancies=True) -> Atoms:
        """Returns an Atoms object from a cif tags dictionary.  See read_cif()
        for a description of the arguments."""
        if primitive_cell and subtrans_included:
            raise RuntimeError('Primitive cell cannot be determined when sublattice translations are included in the symmetry operations listed in the CIF file, i.e. when `subtrans_included` is True.')
        cell = self.get_cell()
        assert cell.rank in [0, 3]
        kwargs: Dict[str, Any] = {}
        if store_tags:
            kwargs['info'] = self._tags.copy()
        if fractional_occupancies:
            occupancies = self._get_fractional_occupancies()
        else:
            occupancies = None
        if occupancies is not None:
            kwargs['onduplicates'] = 'keep'
        unsymmetrized_structure = self.get_unsymmetrized_structure()
        if cell.rank == 3:
            spacegroup = self.get_spacegroup(subtrans_included)
            atoms = crystal(unsymmetrized_structure, spacegroup=spacegroup, setting=spacegroup.setting, occupancies=occupancies, primitive_cell=primitive_cell, **kwargs)
        else:
            atoms = unsymmetrized_structure
            if kwargs.get('info') is not None:
                atoms.info.update(kwargs['info'])
            if occupancies is not None:
                occ_dict = {}
                for i, sym in enumerate(atoms.symbols):
                    occ_dict[str(i)] = {sym: occupancies[i]}
                atoms.info['occupancy'] = occ_dict
        return atoms