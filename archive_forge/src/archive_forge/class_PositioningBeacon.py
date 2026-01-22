from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@implementer(ipositioning.IPositioningBeacon)
class PositioningBeacon:
    """
    A positioning beacon.

    @ivar identifier: The unique identifier for this beacon. This is usually
        an integer. For GPS, this is also known as the PRN.
    @type identifier: Pretty much anything that can be used as a unique
        identifier. Depends on the implementation.
    """

    def __init__(self, identifier):
        """
        Initializes a positioning beacon.

        @param identifier: The identifier for this beacon.
        @type identifier: Can be pretty much anything (see ivar documentation).
        """
        self.identifier = identifier

    def __hash__(self):
        """
        Returns the hash of the identifier for this beacon.

        @return: The hash of the identifier. (C{hash(self.identifier)})
        @rtype: C{int}
        """
        return hash(self.identifier)

    def __repr__(self) -> str:
        """
        Returns a string representation of this beacon.

        @return: The string representation.
        @rtype: C{str}
        """
        return f'<Beacon ({self.identifier})>'