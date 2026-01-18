from typing import Any
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import tostring
Wraps a matcher to define equality in terms of satisfying the matcher.

    ``match_equality`` allows Hamcrest matchers to be used in libraries that
    are not Hamcrest-aware. They might use the equality operator::

        assert match_equality(matcher) == object

    Or they might provide a method that uses equality for its test::

        library.method_that_tests_eq(match_equality(matcher))

    One concrete example is integrating with the ``assert_called_with`` methods
    in Michael Foord's `mock <http://www.voidspace.org.uk/python/mock/>`_
    library.

    