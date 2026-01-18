from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.matcher import Matcher
Matches if object is a string containing a given list of substrings in
    relative order.

    :param string1,...:  A comma-separated list of strings.

    This matcher first checks whether the evaluated object is a string. If so,
    it checks whether it contains a given list of strings, in relative order to
    each other. The searches are performed starting from the beginning of the
    evaluated string.

    Example::

        string_contains_in_order("bc", "fg", "jkl")

    will match "abcdefghijklm".

    