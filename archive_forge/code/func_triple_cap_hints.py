from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def triple_cap_hints(self, hints_info):
    """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Triple cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
    first_cap_index_perfect = self.options['first_cap_index']
    second_cap_index_perfect = self.options['second_cap_index']
    third_cap_index_perfect = self.options['third_cap_index']
    nb_set = hints_info['nb_set']
    permutation = hints_info['permutation']
    nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
    first_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[first_cap_index_perfect]
    second_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[second_cap_index_perfect]
    third_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[third_cap_index_perfect]
    new_site_voronoi_indices1 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices2 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices3 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices4 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices5 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices6 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices7 = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices1.remove(first_cap_voronoi_index)
    new_site_voronoi_indices2.remove(second_cap_voronoi_index)
    new_site_voronoi_indices3.remove(third_cap_voronoi_index)
    new_site_voronoi_indices4.remove(second_cap_voronoi_index)
    new_site_voronoi_indices4.remove(third_cap_voronoi_index)
    new_site_voronoi_indices5.remove(first_cap_voronoi_index)
    new_site_voronoi_indices5.remove(third_cap_voronoi_index)
    new_site_voronoi_indices6.remove(first_cap_voronoi_index)
    new_site_voronoi_indices6.remove(second_cap_voronoi_index)
    new_site_voronoi_indices7.remove(first_cap_voronoi_index)
    new_site_voronoi_indices7.remove(second_cap_voronoi_index)
    new_site_voronoi_indices7.remove(third_cap_voronoi_index)
    return [new_site_voronoi_indices1, new_site_voronoi_indices2, new_site_voronoi_indices3, new_site_voronoi_indices4, new_site_voronoi_indices5, new_site_voronoi_indices6, new_site_voronoi_indices7]