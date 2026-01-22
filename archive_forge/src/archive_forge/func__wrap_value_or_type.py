from typing import Type, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import is_matchable_type, wrap_matcher
from hamcrest.core.matcher import Matcher
from .isinstanceof import instance_of
def _wrap_value_or_type(x):
    if is_matchable_type(x):
        return instance_of(x)
    else:
        return wrap_matcher(x)