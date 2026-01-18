import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
Note that the end index returned by this function is inclusive.
    To use it for Span creation, increment the end by 1.