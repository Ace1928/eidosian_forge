import re
from typing import Any, List, Tuple, Union
from .errors import ParseError

        Assuming a sequence of length n, calculate the start and stop indices,
        and the stride length of the PageRange.

        See help(slice.indices).

        Args:
            n:  the length of the list of pages to choose from.

        Returns:
            Arguments for range()
        