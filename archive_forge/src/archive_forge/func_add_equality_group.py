import collections
from typing import Any, Callable, List, Tuple, Union
import itertools
def add_equality_group(self, *group_items: Any):
    """Tries to add a disjoint equivalence group to the equality tester.

        This methods asserts that items within the group must all be equal to
        each other, but not equal to any items in other groups that have been
        or will be added.

        Args:
          *group_items: The items making up the equivalence group.

        Raises:
            AssertionError: Items within the group are not equal to each other,
                or items in another group are equal to items within the new
                group, or the items violate the equals-implies-same-hash rule.
        """
    self._verify_equality_group(*group_items)
    self._groups.append(group_items)