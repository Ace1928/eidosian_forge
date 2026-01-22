from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform

        Create a new evolver for this pmap. For a discussion on evolvers in general see the
        documentation for the pvector evolver.

        Create the evolver and perform various mutating updates to it:

        >>> m1 = m(a=1, b=2)
        >>> e = m1.evolver()
        >>> e['c'] = 3
        >>> len(e)
        3
        >>> del e['a']

        The underlying pmap remains the same:

        >>> m1 == {'a': 1, 'b': 2}
        True

        The changes are kept in the evolver. An updated pmap can be created using the
        persistent() function on the evolver.

        >>> m2 = e.persistent()
        >>> m2 == {'b': 2, 'c': 3}
        True

        The new pmap will share data with the original pmap in the same way that would have
        been done if only using operations on the pmap.
        