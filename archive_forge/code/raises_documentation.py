import re
import sys
from typing import Any, Callable, Mapping, Optional, Tuple, Type, cast
from weakref import ref
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
Wrapper for function call that delays the actual execution so that
    :py:func:`~hamcrest.core.core.raises.raises` matcher can catch any thrown exception.

    :param func: The function or method to be called

    The arguments can be provided with a call to the `with_args` function on the returned
    object::

           calling(my_method).with_args(arguments, and_='keywords')
    