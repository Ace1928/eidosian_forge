from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
class NeighborsSet:
    """
        Class used to store a given set of neighbors of a given site (based on a list of sites, the voronoi
        container is not part of the LightStructureEnvironments object).
        """

    def __init__(self, structure: Structure, isite, all_nbs_sites, all_nbs_sites_indices):
        """Constructor for NeighborsSet.

            Args:
                structure: Structure object.
                isite: Index of the site for which neighbors are stored in this NeighborsSet.
                all_nbs_sites: All the possible neighbors for this site.
                all_nbs_sites_indices: Indices of the sites in all_nbs_sites that make up this NeighborsSet.
            """
        self.structure = structure
        self.isite = isite
        self.all_nbs_sites = all_nbs_sites
        indices = set(all_nbs_sites_indices)
        if len(indices) != len(all_nbs_sites_indices):
            raise ValueError('Set of neighbors contains duplicates !')
        self.all_nbs_sites_indices = sorted(indices)
        self.all_nbs_sites_indices_unsorted = all_nbs_sites_indices

    @property
    def neighb_coords(self):
        """Coordinates of neighbors for this NeighborsSet."""
        return [self.all_nbs_sites[inb]['site'].coords for inb in self.all_nbs_sites_indices_unsorted]

    @property
    def neighb_sites(self):
        """Neighbors for this NeighborsSet as pymatgen Sites."""
        return [self.all_nbs_sites[inb]['site'] for inb in self.all_nbs_sites_indices_unsorted]

    @property
    def neighb_sites_and_indices(self):
        """List of neighbors for this NeighborsSet as pymatgen Sites and their index in the original structure."""
        return [{'site': self.all_nbs_sites[inb]['site'], 'index': self.all_nbs_sites[inb]['index']} for inb in self.all_nbs_sites_indices_unsorted]

    @property
    def neighb_indices_and_images(self) -> list[dict[str, int]]:
        """List of indices and images with respect to the original unit cell sites for this NeighborsSet."""
        return [{'index': self.all_nbs_sites[inb]['index'], 'image_cell': self.all_nbs_sites[inb]['image_cell']} for inb in self.all_nbs_sites_indices_unsorted]

    def __len__(self) -> int:
        return len(self.all_nbs_sites_indices)

    def __hash__(self) -> int:
        return len(self.all_nbs_sites_indices)

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('isite', 'all_nbs_sites_indices')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs))

    def __str__(self):
        return f'Neighbors Set for site #{self.isite} :\n - Coordination number : {len(self)}\n - Neighbors sites indices : {', '.join((f'{nb_idxs}' for nb_idxs in self.all_nbs_sites_indices))}\n'

    def as_dict(self):
        """A JSON-serializable dict representation of the NeighborsSet."""
        return {'isite': self.isite, 'all_nbs_sites_indices': self.all_nbs_sites_indices_unsorted}

    @classmethod
    def from_dict(cls, dct, structure: Structure, all_nbs_sites) -> Self:
        """
            Reconstructs the NeighborsSet algorithm from its JSON-serializable dict representation, together with
            the structure and all the possible neighbors sites.

            As an inner (nested) class, the NeighborsSet is not supposed to be used anywhere else that inside the
            LightStructureEnvironments. The from_dict method is thus using the structure and all_nbs_sites when
            reconstructing itself. These two are both in the LightStructureEnvironments object.

            Args:
                dct: a JSON-serializable dict representation of a NeighborsSet.
                structure: The structure.
                all_nbs_sites: The list of all the possible neighbors for a given site.

            Returns:
                NeighborsSet
            """
        return cls(structure=structure, isite=dct['isite'], all_nbs_sites=all_nbs_sites, all_nbs_sites_indices=dct['all_nbs_sites_indices'])