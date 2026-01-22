from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
Create a `KeysEqual` Matcher.

        :param expected: The keys the matchee is expected to have. As a
            special case, if a single argument is specified, and it is a
            mapping, then we use its keys as the expected set.
        