from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
Matches if object is a string equal to a given string, ignoring case
    differences.

    :param string: The string to compare against as the expected value.

    This matcher first checks whether the evaluated object is a string. If so,
    it compares it with ``string``, ignoring differences of case.

    Example::

        equal_to_ignoring_case("hello world")

    will match "heLLo WorlD".

    