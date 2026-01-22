from __future__ import annotations
import abc
from monty.json import MSONable
class EnvironmentNode(AbstractEnvironmentNode):
    """Class used to define an environment as a node in a graph."""

    def __init__(self, central_site, i_central_site, ce_symbol) -> None:
        """
        Constructor for the EnvironmentNode object.

        Args:
            central_site (Site or subclass of Site): central site as a pymatgen Site or
                subclass of Site (e.g. PeriodicSite, ...).
            i_central_site (int): Index of the central site in the structure.
            ce_symbol (str): Symbol of the identified environment.
        """
        AbstractEnvironmentNode.__init__(self, central_site, i_central_site)
        self._ce_symbol = ce_symbol

    @property
    def coordination_environment(self):
        """Coordination environment of this node."""
        return self._ce_symbol

    def everything_equal(self, other):
        """Compare with another environment node.

        Returns:
            bool: True if it is equal to the other node.
        """
        return super().everything_equal(other) and self.coordination_environment == other.coordination_environment