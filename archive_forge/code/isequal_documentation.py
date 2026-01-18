from typing import Any
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
Matches if object is equal to a given object.

    :param obj: The object to compare against as the expected value.

    This matcher compares the evaluated object to ``obj`` for equality.