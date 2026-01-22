import collections
from typing import Any, Callable, List, Tuple, Union
import itertools
Tries to add a disjoint equivalence group to the equality tester.

        Uses the factory methods to produce two different objects with the same
        initialization for each factory. Asserts that the objects are equal, but
        not equal to any items in other groups that have been or will be added.
        Adds the objects as a group.

        Args:
            *factories: Methods for producing independent copies of an item.

        Raises:
            AssertionError: The factories produce items not equal to the others,
                or items in another group are equal to items from the factory,
                or the items violate the equal-implies-same-hash rule.
        