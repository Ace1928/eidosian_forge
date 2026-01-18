from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.matcher import Matcher
from hamcrest.library.text.substringmatcher import SubstringMatcher
Matches if object is a string ending with a given string.

    :param string: The string to search for.

    This matcher first checks whether the evaluated object is a string. If so,
    it checks if ``string`` matches the ending characters of the evaluated
    object.

    Example::

        ends_with("bar")

    will match "foobar".

    