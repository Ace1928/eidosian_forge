import collections
from typing import Any, Callable, List, Tuple, Union
import itertools
class EqualsTester:
    """Tests equality against user-provided disjoint equivalence groups."""

    def __init__(self) -> None:
        self._groups: List[Tuple[Union[Any, _ClassUnknownToSubjects], ...]] = [(_ClassUnknownToSubjects(),)]

    def _verify_equality_group(self, *group_items: Any):
        """Verifies that a group is an equivalence group.

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
        assert group_items
        for v1, v2 in itertools.product(group_items, group_items):
            same = _eq_check(v1, v2)
            assert same or v1 is not v2, f"{v1!r} isn't equal to itself!"
            assert same, f"{v1!r} and {v2!r} can't be in the same equality group. They're not equal."
        for other_group in self._groups:
            for v1, v2 in itertools.product(group_items, other_group):
                assert not _eq_check(v1, v2), f"{v1!r} and {v2!r} can't be in different equality groups. They're equal."
        hashes = [hash(v) if isinstance(v, collections.abc.Hashable) else None for v in group_items]
        if len(set(hashes)) > 1:
            examples = ((v1, h1, v2, h2) for v1, h1 in zip(group_items, hashes) for v2, h2 in zip(group_items, hashes) if h1 != h2)
            example = next(examples)
            raise AssertionError(f'Items in the same group produced different hashes. Example: hash({example[0]!r}) is {example[1]!r} but hash({example[2]!r}) is {example[3]!r}.')
        for v in group_items:
            assert _TestsForNotImplemented(v) == v and v == _TestsForNotImplemented(v), f'An item did not return NotImplemented when checking equality of this item against a different type than the item. Relevant item: {v!r}. Common problem: returning NotImplementedError instead of NotImplemented. '

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

    def make_equality_group(self, *factories: Callable[[], Any]):
        """Tries to add a disjoint equivalence group to the equality tester.

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
        """
        self.add_equality_group(*(f() for f in factories for _ in range(2)))