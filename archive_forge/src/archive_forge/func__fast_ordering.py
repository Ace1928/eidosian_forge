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
def _fast_ordering(self, structure: Structure, num_remove_dict, num_to_return=1):
    """This method uses the matrix form of Ewald sum to calculate the ewald
        sums of the potential structures. This is on the order of 4 orders of
        magnitude faster when there are large numbers of permutations to
        consider. There are further optimizations possible (doing a smarter
        search of permutations for example), but this won't make a difference
        until the number of permutations is on the order of 30,000.
        """
    self.logger.debug('Performing fast ordering')
    start_time = time.perf_counter()
    self.logger.debug('Performing initial Ewald sum...')
    ewald_matrix = EwaldSummation(structure).total_energy_matrix
    self.logger.debug(f'Ewald sum took {time.perf_counter() - start_time} seconds.')
    start_time = time.perf_counter()
    m_list = [[0, num, list(indices), None] for indices, num in num_remove_dict.items()]
    self.logger.debug('Calling EwaldMinimizer...')
    minimizer = EwaldMinimizer(ewald_matrix, m_list, num_to_return, PartialRemoveSitesTransformation.ALGO_FAST)
    self.logger.debug(f'Minimizing Ewald took {time.perf_counter() - start_time} seconds.')
    all_structures = []
    lowest_energy = minimizer.output_lists[0][0]
    num_atoms = sum(structure.composition.values())
    for output in minimizer.output_lists:
        struct = structure.copy()
        del_indices = []
        for manipulation in output[1]:
            if manipulation[1] is None:
                del_indices.append(manipulation[0])
            else:
                struct.replace(manipulation[0], manipulation[1])
        struct.remove_sites(del_indices)
        struct = struct.get_sorted_structure()
        e_above_min = (output[0] - lowest_energy) / num_atoms
        all_structures.append({'energy': output[0], 'energy_above_minimum': e_above_min, 'structure': struct})
    return all_structures