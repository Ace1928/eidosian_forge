from typing import Tuple
import pytest
from nltk.metrics.distance import edit_distance
class TestEditDistance:

    @pytest.mark.parametrize('left,right,substitution_cost,expecteds', [('abc', 'ca', 1, (2, 3)), ('abc', 'ca', 5, (2, 3)), ('wants', 'wasp', 1, (3, 3)), ('wants', 'wasp', 5, (3, 3)), ('rain', 'shine', 1, (3, 3)), ('rain', 'shine', 2, (5, 5)), ('acbdef', 'abcdef', 1, (1, 2)), ('acbdef', 'abcdef', 2, (1, 2)), ('lnaguaeg', 'language', 1, (2, 4)), ('lnaguaeg', 'language', 2, (2, 4)), ('lnaugage', 'language', 1, (2, 3)), ('lnaugage', 'language', 2, (2, 4)), ('lngauage', 'language', 1, (2, 2)), ('lngauage', 'language', 2, (2, 2)), ('wants', 'swim', 1, (5, 5)), ('wants', 'swim', 2, (6, 7)), ('kitten', 'sitting', 1, (3, 3)), ('kitten', 'sitting', 2, (5, 5)), ('duplicated', 'duuplicated', 1, (1, 1)), ('duplicated', 'duuplicated', 2, (1, 1)), ('very duplicated', 'very duuplicateed', 2, (2, 2))])
    def test_with_transpositions(self, left: str, right: str, substitution_cost: int, expecteds: Tuple[int, int]):
        """
        Test `edit_distance` between two strings, given some `substitution_cost`,
        and whether transpositions are allowed.

        :param str left: First input string to `edit_distance`.
        :param str right: Second input string to `edit_distance`.
        :param int substitution_cost: The cost of a substitution action in `edit_distance`.
        :param Tuple[int, int] expecteds: A tuple of expected outputs, such that `expecteds[0]` is
            the expected output with `transpositions=True`, and `expecteds[1]` is
            the expected output with `transpositions=False`.
        """
        for s1, s2 in ((left, right), (right, left)):
            for expected, transpositions in zip(expecteds, [True, False]):
                predicted = edit_distance(s1, s2, substitution_cost=substitution_cost, transpositions=transpositions)
                assert predicted == expected