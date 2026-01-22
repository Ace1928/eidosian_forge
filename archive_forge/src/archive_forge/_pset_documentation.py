from collections.abc import Set, Hashable
import sys
from typing import TypeVar, Generic
from pyrsistent._pmap import pmap

        Create a new evolver for this pset. For a discussion on evolvers in general see the
        documentation for the pvector evolver.

        Create the evolver and perform various mutating updates to it:

        >>> s1 = s(1, 2, 3)
        >>> e = s1.evolver()
        >>> _ = e.add(4)
        >>> len(e)
        4
        >>> _ = e.remove(1)

        The underlying pset remains the same:

        >>> s1
        pset([1, 2, 3])

        The changes are kept in the evolver. An updated pmap can be created using the
        persistent() function on the evolver.

        >>> s2 = e.persistent()
        >>> s2
        pset([2, 3, 4])

        The new pset will share data with the original pset in the same way that would have
        been done if only using operations on the pset.
        