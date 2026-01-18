from typing import Sequence, TypeVar
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
Matches if evaluated object is present in a given sequence.

    :param sequence: The sequence to search.

    This matcher invokes the ``in`` membership operator to determine if the
    evaluated object is a member of the sequence.

    