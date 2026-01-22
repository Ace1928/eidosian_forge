from typing import Any
from cirq.testing.equals_tester import EqualsTester
def add_ascending_equivalence_group(self, *group_items: Any):
    """Tries to add an ascending equivalence group to the order tester.

        Asserts that the group items are equal to each other, but strictly
        ascending with regard to the already added groups.

        Adds the objects as a group.

        Args:
            *group_items: items making the equivalence group

        Raises:
            AssertionError: The group elements aren't equal to each other,
                or items in another group overlap with the new group.
        """
    for item in group_items:
        self._verify_not_implemented_vs_unknown(item)
    for item1 in group_items:
        for item2 in group_items:
            self._verify_ordering(item1, item2, 0)
    for lesser_group in self._groups:
        for lesser_item in lesser_group:
            for larger_item in group_items:
                self._verify_ordering(lesser_item, larger_item, +1)
    self._eq_tester.add_equality_group(*group_items)
    self._groups.append(group_items)