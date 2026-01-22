from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
def _generate_transformations(self, structure: Structure) -> dict[str, MagOrderingTransformation]:
    """The central problem with trying to enumerate magnetic orderings is
        that we have to enumerate orderings that might plausibly be magnetic
        ground states, while not enumerating orderings that are physically
        implausible. The problem is that it is not always obvious by e.g.
        symmetry arguments alone which orderings to prefer. Here, we use a
        variety of strategies (heuristics) to enumerate plausible orderings,
        and later discard any duplicates that might be found by multiple
        strategies. This approach is not ideal, but has been found to be
        relatively robust over a wide range of magnetic structures.

        Args:
            structure: A sanitized input structure (_sanitize_input_structure)

        Returns:
            dict: A dict of a transformation class instance (values) and name of enumeration strategy (keys)
        """
    formula = structure.reduced_formula
    transformations: dict[str, MagOrderingTransformation] = {}
    analyzer = CollinearMagneticStructureAnalyzer(structure, default_magmoms=self.default_magmoms, overwrite_magmom_mode='replace_all')
    if not analyzer.is_magnetic:
        raise ValueError('Not detected as magnetic, add a new default magmom for the element you believe may be magnetic?')
    self.logger.info(f'Generating magnetic orderings for {formula}')
    mag_species_spin = analyzer.magnetic_species_and_magmoms
    types_mag_species = sorted(analyzer.types_of_magnetic_species, key=lambda sp: analyzer.default_magmoms.get(str(sp), 0), reverse=True)
    num_mag_sites = analyzer.number_of_magnetic_sites
    num_unique_sites = analyzer.number_of_unique_magnetic_sites()
    if num_unique_sites > self.max_unique_sites:
        raise ValueError('Too many magnetic sites to sensibly perform enumeration.')
    if 'max_cell_size' not in self.transformation_kwargs:
        self.transformation_kwargs['max_cell_size'] = max(1, int(4 / num_mag_sites))
    self.logger.info(f'Max cell size set to {self.transformation_kwargs['max_cell_size']}')
    sga = SpacegroupAnalyzer(structure)
    structure_sym = sga.get_symmetrized_structure()
    wyckoff = ['n/a'] * len(structure)
    for indices, symbol in zip(structure_sym.equivalent_indices, structure_sym.wyckoff_symbols):
        for index in indices:
            wyckoff[index] = symbol
    is_magnetic_sites = [site.specie in types_mag_species for site in structure]
    wyckoff = [symbol if is_magnetic_site else 'n/a' for symbol, is_magnetic_site in zip(wyckoff, is_magnetic_sites)]
    structure.add_site_property('wyckoff', wyckoff)
    wyckoff_symbols = set(wyckoff) - {'n/a'}
    if self.automatic:
        if 'ferrimagnetic_by_motif' not in self.strategies and len(wyckoff_symbols) > 1 and (len(types_mag_species) == 1):
            self.strategies += ['ferrimagnetic_by_motif']
        if 'antiferromagnetic_by_motif' not in self.strategies and len(wyckoff_symbols) > 1 and (len(types_mag_species) == 1):
            self.strategies += ['antiferromagnetic_by_motif']
        if 'ferrimagnetic_by_species' not in self.strategies and len(types_mag_species) > 1:
            self.strategies += ['ferrimagnetic_by_species']
    if 'ferromagnetic' in self.strategies:
        fm_structure = analyzer.get_ferromagnetic_structure()
        fm_structure.add_spin_by_site(fm_structure.site_properties['magmom'])
        fm_structure.remove_site_property('magmom')
        self.ordered_structures.append(fm_structure)
        self.ordered_structure_origins.append('fm')
    all_constraints: dict[str, Any] = {}
    if 'antiferromagnetic' in self.strategies:
        constraint = MagOrderParameterConstraint(0.5, species_constraints=list(map(str, types_mag_species)))
        all_constraints['afm'] = [constraint]
        if len(types_mag_species) > 1:
            for sp in types_mag_species:
                constraints = [MagOrderParameterConstraint(0.5, species_constraints=str(sp))]
                all_constraints[f'afm_by_{sp}'] = constraints
    if 'ferrimagnetic_by_motif' in self.strategies and len(wyckoff_symbols) > 1:
        for symbol in wyckoff_symbols:
            constraints = [MagOrderParameterConstraint(0.5, site_constraint_name='wyckoff', site_constraints=symbol), MagOrderParameterConstraint(1, site_constraint_name='wyckoff', site_constraints=list(wyckoff_symbols - {symbol}))]
            all_constraints[f'ferri_by_motif_{symbol}'] = constraints
    if 'ferrimagnetic_by_species' in self.strategies:
        sp_list = [str(site.specie) for site in structure]
        num_sp = {sp: sp_list.count(str(sp)) for sp in types_mag_species}
        total_mag_sites = sum(num_sp.values())
        for sp in types_mag_species:
            all_constraints[f'ferri_by_{sp}'] = num_sp[sp] / total_mag_sites
            constraints = [MagOrderParameterConstraint(0.5, species_constraints=str(sp)), MagOrderParameterConstraint(1, species_constraints=list(map(str, set(types_mag_species) - {sp})))]
            all_constraints[f'ferri_by_{sp}_afm'] = constraints
    if 'antiferromagnetic_by_motif' in self.strategies:
        for symbol in wyckoff_symbols:
            constraints = [MagOrderParameterConstraint(0.5, site_constraint_name='wyckoff', site_constraints=symbol)]
            all_constraints[f'afm_by_motif_{symbol}'] = constraints
    transformations = {}
    for name, constraints in all_constraints.items():
        trans = MagOrderingTransformation(mag_species_spin, order_parameter=constraints, **self.transformation_kwargs)
        transformations[name] = trans
    return transformations