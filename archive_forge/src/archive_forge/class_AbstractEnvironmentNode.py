from __future__ import annotations
import abc
from monty.json import MSONable
class AbstractEnvironmentNode(MSONable):
    """Abstract class used to define an environment as a node in a graph."""
    COORDINATION_ENVIRONMENT = 0
    NUMBER_OF_NEIGHBORING_COORDINATION_ENVIRONMENTS = 1
    NUMBER_OF_NEIGHBORING_CES = NUMBER_OF_NEIGHBORING_COORDINATION_ENVIRONMENTS
    NEIGHBORING_COORDINATION_ENVIRONMENTS = 2
    NEIGHBORING_CES = NEIGHBORING_COORDINATION_ENVIRONMENTS
    NUMBER_OF_LIGANDS_FOR_EACH_NEIGHBORING_COORDINATION_ENVIRONMENT = 3
    NUMBER_OF_LIGANDS_FOR_EACH_NEIGHBORING_CE = NUMBER_OF_LIGANDS_FOR_EACH_NEIGHBORING_COORDINATION_ENVIRONMENT
    LIGANDS_ARRANGEMENT = 4
    NEIGHBORS_LIGANDS_ARRANGEMENT = 5
    ATOM = 6
    CE_NNBCES_NBCES_LIGANDS = -1
    DEFAULT_EXTENSIONS = (ATOM, COORDINATION_ENVIRONMENT)

    def __init__(self, central_site, i_central_site) -> None:
        """
        Constructor for the AbstractEnvironmentNode object.

        Args:
            central_site (Site or subclass of Site): central site as a pymatgen Site or
                subclass of Site (e.g. PeriodicSite, ...).
            i_central_site (int): Index of the central site in the structure.
        """
        self.central_site = central_site
        self.i_central_site = i_central_site

    @property
    def isite(self):
        """Index of the central site."""
        return self.i_central_site

    def __hash__(self) -> int:
        """Simple hash function based on the hash function of the central site."""
        return hash(self.central_site)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractEnvironmentNode):
            return NotImplemented
        return self.__class__ == other.__class__ and self.isite == other.isite

    def __lt__(self, other):
        return self.isite < other.isite

    def everything_equal(self, other):
        """Checks equality with respect to another AbstractEnvironmentNode using the index of the central site
        as well as the central site itself.
        """
        return self == other and self.central_site == other.central_site

    @property
    @abc.abstractmethod
    def coordination_environment(self):
        """Coordination environment of this node."""
        return

    @property
    def ce(self):
        """Coordination environment of this node."""
        return self.coordination_environment

    @property
    def mp_symbol(self):
        """Coordination environment of this node."""
        return self.coordination_environment

    @property
    def ce_symbol(self):
        """Coordination environment of this node."""
        return self.coordination_environment

    @property
    def atom_symbol(self):
        """Symbol of the atom on the central site."""
        return self.central_site.specie.symbol

    def __str__(self):
        """String representation of the AbstractEnvironmentNode."""
        return f'Node #{self.isite} {self.atom_symbol} ({self.coordination_environment})'