from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
class CifWriter:
    """A wrapper around CifFile to write CIF files from pymatgen structures."""

    def __init__(self, struct: Structure, symprec: float | None=None, write_magmoms: bool=False, significant_figures: int=8, angle_tolerance: float=5, refine_struct: bool=True, write_site_properties: bool=False) -> None:
        """
        Args:
            struct (Structure): structure to write
            symprec (float): If not none, finds the symmetry of the structure
                and writes the cif with symmetry information. Passes symprec
                to the SpacegroupAnalyzer. See also refine_struct.
            write_magmoms (bool): If True, will write magCIF file. Incompatible
                with symprec
            significant_figures (int): Specifies precision for formatting of floats.
                Defaults to 8.
            angle_tolerance (float): Angle tolerance for symmetry finding. Passes
                angle_tolerance to the SpacegroupAnalyzer. Used only if symprec
                is not None.
            refine_struct: Used only if symprec is not None. If True, get_refined_structure
                is invoked to convert input structure from primitive to conventional.
            write_site_properties (bool): Whether to write the Structure.site_properties
                to the CIF as _atom_site_{property name}. Defaults to False.
        """
        if write_magmoms and symprec:
            warnings.warn('Magnetic symmetry cannot currently be detected by pymatgen,disabling symmetry detection.')
            symprec = None
        format_str = f'{{:.{significant_figures}f}}'
        block: dict[str, Any] = {}
        loops = []
        spacegroup = ('P 1', 1)
        if symprec is not None:
            spg_analyzer = SpacegroupAnalyzer(struct, symprec, angle_tolerance=angle_tolerance)
            spacegroup = (spg_analyzer.get_space_group_symbol(), spg_analyzer.get_space_group_number())
            if refine_struct:
                struct = spg_analyzer.get_refined_structure()
        lattice = struct.lattice
        comp = struct.composition
        no_oxi_comp = comp.element_composition
        block['_symmetry_space_group_name_H-M'] = spacegroup[0]
        for cell_attr in ['a', 'b', 'c']:
            block['_cell_length_' + cell_attr] = format_str.format(getattr(lattice, cell_attr))
        for cell_attr in ['alpha', 'beta', 'gamma']:
            block['_cell_angle_' + cell_attr] = format_str.format(getattr(lattice, cell_attr))
        block['_symmetry_Int_Tables_number'] = spacegroup[1]
        block['_chemical_formula_structural'] = no_oxi_comp.reduced_formula
        block['_chemical_formula_sum'] = no_oxi_comp.formula
        block['_cell_volume'] = format_str.format(lattice.volume)
        _, fu = no_oxi_comp.get_reduced_composition_and_factor()
        block['_cell_formula_units_Z'] = str(int(fu))
        if symprec is None:
            block['_symmetry_equiv_pos_site_id'] = ['1']
            block['_symmetry_equiv_pos_as_xyz'] = ['x, y, z']
        else:
            spg_analyzer = SpacegroupAnalyzer(struct, symprec)
            symm_ops: list[SymmOp] = []
            for op in spg_analyzer.get_symmetry_operations():
                v = op.translation_vector
                symm_ops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))
            ops = [op.as_xyz_str() for op in symm_ops]
            block['_symmetry_equiv_pos_site_id'] = [f'{i}' for i in range(1, len(ops) + 1)]
            block['_symmetry_equiv_pos_as_xyz'] = ops
        loops.append(['_symmetry_equiv_pos_site_id', '_symmetry_equiv_pos_as_xyz'])
        try:
            symbol_to_oxi_num = {str(el): float(el.oxi_state or 0) for el in sorted(comp.elements)}
            block['_atom_type_symbol'] = list(symbol_to_oxi_num)
            block['_atom_type_oxidation_number'] = symbol_to_oxi_num.values()
            loops.append(['_atom_type_symbol', '_atom_type_oxidation_number'])
        except (TypeError, AttributeError):
            symbol_to_oxi_num = {el.symbol: 0 for el in sorted(comp.elements)}
        atom_site_type_symbol = []
        atom_site_symmetry_multiplicity = []
        atom_site_fract_x = []
        atom_site_fract_y = []
        atom_site_fract_z = []
        atom_site_label = []
        atom_site_occupancy = []
        atom_site_moment_label = []
        atom_site_moment_crystalaxis_x = []
        atom_site_moment_crystalaxis_y = []
        atom_site_moment_crystalaxis_z = []
        atom_site_properties: dict[str, list] = defaultdict(list)
        count = 0
        if symprec is None:
            for site in struct:
                for sp, occu in sorted(site.species.items()):
                    atom_site_type_symbol.append(str(sp))
                    atom_site_symmetry_multiplicity.append('1')
                    atom_site_fract_x.append(format_str.format(site.a))
                    atom_site_fract_y.append(format_str.format(site.b))
                    atom_site_fract_z.append(format_str.format(site.c))
                    atom_site_occupancy.append(str(occu))
                    site_label = f'{sp.symbol}{count}'
                    if 'magmom' in site.properties:
                        mag = site.properties['magmom']
                    elif getattr(sp, 'spin', None) is not None:
                        mag = sp.spin
                    else:
                        site_label = site.label if site.label != site.species_string else site_label
                        mag = 0
                    atom_site_label.append(site_label)
                    magmom = Magmom(mag)
                    if write_magmoms and abs(magmom) > 0:
                        moment = Magmom.get_moment_relative_to_crystal_axes(magmom, lattice)
                        atom_site_moment_label.append(f'{sp.symbol}{count}')
                        atom_site_moment_crystalaxis_x.append(format_str.format(moment[0]))
                        atom_site_moment_crystalaxis_y.append(format_str.format(moment[1]))
                        atom_site_moment_crystalaxis_z.append(format_str.format(moment[2]))
                    if write_site_properties:
                        for key, val in site.properties.items():
                            atom_site_properties[key].append(format_str.format(val))
                    count += 1
        else:
            unique_sites = [(sorted(sites, key=lambda s: tuple((abs(x) for x in s.frac_coords)))[0], len(sites)) for sites in spg_analyzer.get_symmetrized_structure().equivalent_sites]
            for site, mult in sorted(unique_sites, key=lambda t: (t[0].species.average_electroneg, -t[1], t[0].a, t[0].b, t[0].c)):
                for sp, occu in site.species.items():
                    atom_site_type_symbol.append(str(sp))
                    atom_site_symmetry_multiplicity.append(f'{mult}')
                    atom_site_fract_x.append(format_str.format(site.a))
                    atom_site_fract_y.append(format_str.format(site.b))
                    atom_site_fract_z.append(format_str.format(site.c))
                    site_label = site.label if site.label != site.species_string else f'{sp.symbol}{count}'
                    atom_site_label.append(site_label)
                    atom_site_occupancy.append(str(occu))
                    count += 1
        block['_atom_site_type_symbol'] = atom_site_type_symbol
        block['_atom_site_label'] = atom_site_label
        block['_atom_site_symmetry_multiplicity'] = atom_site_symmetry_multiplicity
        block['_atom_site_fract_x'] = atom_site_fract_x
        block['_atom_site_fract_y'] = atom_site_fract_y
        block['_atom_site_fract_z'] = atom_site_fract_z
        block['_atom_site_occupancy'] = atom_site_occupancy
        loop_labels = ['_atom_site_type_symbol', '_atom_site_label', '_atom_site_symmetry_multiplicity', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z', '_atom_site_occupancy']
        if write_site_properties:
            for key, vals in atom_site_properties.items():
                block[f'_atom_site_{key}'] = vals
                loop_labels += [f'_atom_site_{key}']
        loops.append(loop_labels)
        if write_magmoms:
            block['_atom_site_moment_label'] = atom_site_moment_label
            block['_atom_site_moment_crystalaxis_x'] = atom_site_moment_crystalaxis_x
            block['_atom_site_moment_crystalaxis_y'] = atom_site_moment_crystalaxis_y
            block['_atom_site_moment_crystalaxis_z'] = atom_site_moment_crystalaxis_z
            loops.append(['_atom_site_moment_label', '_atom_site_moment_crystalaxis_x', '_atom_site_moment_crystalaxis_y', '_atom_site_moment_crystalaxis_z'])
        dct = {}
        dct[comp.reduced_formula] = CifBlock(block, loops, comp.reduced_formula)
        self._cf = CifFile(dct)

    @property
    def cif_file(self):
        """Returns: CifFile associated with the CifWriter."""
        return self._cf

    def __str__(self):
        """Returns the CIF as a string."""
        return str(self._cf)

    def write_file(self, filename: str | Path, mode: Literal['w', 'a', 'wt', 'at']='w') -> None:
        """Write the CIF file."""
        with zopen(filename, mode=mode) as file:
            file.write(str(self))