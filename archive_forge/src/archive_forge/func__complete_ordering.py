from __future__ import annotations
import itertools
import logging
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.transformation_abc import AbstractTransformation
def _complete_ordering(self, structure: Structure, num_remove_dict):
    self.logger.debug('Performing complete ordering...')
    all_structures: list[dict[str, float | Structure]] = []
    symprec = 0.2
    spg_analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    self.logger.debug(f'Symmetry of structure is determined to be {spg_analyzer.get_space_group_symbol()}.')
    sg = spg_analyzer.get_space_group_operations()
    tested_sites: list[list[PeriodicSite]] = []
    start_time = time.perf_counter()
    self.logger.debug('Performing initial Ewald sum...')
    ewald_sum = EwaldSummation(structure)
    self.logger.debug(f'Ewald sum took {time.perf_counter() - start_time} seconds.')
    start_time = time.perf_counter()
    all_combis = [list(itertools.combinations(ind, num)) for ind, num in num_remove_dict.items()]
    for idx, all_indices in enumerate(itertools.product(*all_combis), 1):
        sites_to_remove = []
        indices_list = []
        for indices in all_indices:
            sites_to_remove.extend([structure[i] for i in indices])
            indices_list.extend(indices)
        s_new = structure.copy()
        s_new.remove_sites(indices_list)
        energy = ewald_sum.compute_partial_energy(indices_list)
        already_tested = False
        for ii, t_sites in enumerate(tested_sites):
            t_energy = all_structures[ii]['energy']
            if abs((energy - t_energy) / len(s_new)) < 1e-05 and sg.are_symmetrically_equivalent(sites_to_remove, t_sites, symm_prec=symprec):
                already_tested = True
        if not already_tested:
            tested_sites.append(sites_to_remove)
            all_structures.append({'structure': s_new, 'energy': energy})
        if idx % 10 == 0:
            now = time.perf_counter()
            self.logger.debug(f'{idx} structures, {now - start_time:.2f} seconds.')
            self.logger.debug(f'Average time per combi = {(now - start_time) / idx} seconds')
            self.logger.debug(f'{len(all_structures)} symmetrically distinct structures found.')
    self.logger.debug(f'Total symmetrically distinct structures found = {len(all_structures)}')
    return sorted(all_structures, key=lambda s: s['energy'])