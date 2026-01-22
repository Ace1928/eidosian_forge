from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
class SubstrateAnalyzer(ZSLGenerator):
    """
    This class applies a set of search criteria to identify suitable
    substrates for film growth. It first uses a topological search by Zur
    and McGill to identify matching super-lattices on various faces of the
    two materials. Additional criteria can then be used to identify the most
    suitable substrate. Currently, the only additional criteria is the
    elastic strain energy of the super-lattices.
    """

    def __init__(self, film_max_miller=1, substrate_max_miller=1, **kwargs):
        """
        Initializes the substrate analyzer

        Args:
            zslgen (ZSLGenerator): Defaults to a ZSLGenerator with standard
                tolerances, but can be fed one with custom tolerances
            film_max_miller (int): maximum miller index to generate for film
                surfaces
            substrate_max_miller (int): maximum miller index to generate for
                substrate surfaces.
        """
        self.film_max_miller = film_max_miller
        self.substrate_max_miller = substrate_max_miller
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def generate_surface_vectors(self, film: Structure, substrate: Structure, film_millers: ArrayLike, substrate_millers: ArrayLike):
        """
        Generates the film/substrate slab combinations for a set of given
        miller indices.

        Args:
            film (Structure): film structure
            substrate (Structure): substrate structure
            film_millers (array): all miller indices to generate slabs for
                film
            substrate_millers (array): all miller indices to generate slabs
                for substrate
        """
        vector_sets = []
        for f_miller in film_millers:
            film_slab = SlabGenerator(film, f_miller, 20, 15, primitive=False).get_slab()
            film_vectors = reduce_vectors(film_slab.oriented_unit_cell.lattice.matrix[0], film_slab.oriented_unit_cell.lattice.matrix[1])
            for s_miller in substrate_millers:
                substrate_slab = SlabGenerator(substrate, s_miller, 20, 15, primitive=False).get_slab()
                substrate_vectors = reduce_vectors(substrate_slab.oriented_unit_cell.lattice.matrix[0], substrate_slab.oriented_unit_cell.lattice.matrix[1])
                vector_sets.append((film_vectors, substrate_vectors, f_miller, s_miller))
        return vector_sets

    def calculate(self, film: Structure, substrate: Structure, elasticity_tensor=None, film_millers: ArrayLike=None, substrate_millers: ArrayLike=None, ground_state_energy=0, lowest=False):
        """
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
        """
        if film_millers is None:
            film_millers = sorted(get_symmetrically_distinct_miller_indices(film, self.film_max_miller))
        if substrate_millers is None:
            substrate_millers = sorted(get_symmetrically_distinct_miller_indices(substrate, self.substrate_max_miller))
        surface_vector_sets = self.generate_surface_vectors(film, substrate, film_millers, substrate_millers)
        for [film_vectors, substrate_vectors, film_miller, substrate_miller] in surface_vector_sets:
            for match in self(film_vectors, substrate_vectors, lowest):
                sub_match = SubstrateMatch.from_zsl(match=match, film=film, film_miller=film_miller, substrate_miller=substrate_miller, elasticity_tensor=elasticity_tensor, ground_state_energy=ground_state_energy)
                yield sub_match