from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices

        Finds all topological matches for the substrate and calculates elastic
        strain energy and total energy for the film if elasticity tensor and
        ground state energy are provided:

        Args:
            film (Structure): conventional standard structure for the film
            substrate (Structure): conventional standard structure for the
                substrate
            elasticity_tensor (ElasticTensor): elasticity tensor for the film
                in the IEEE orientation
            film_millers (array): film facets to consider in search as defined by
                miller indices
            substrate_millers (array): substrate facets to consider in search as
                defined by miller indices
            ground_state_energy (float): ground state energy for the film
            lowest (bool): only consider lowest matching area for each surface
        