from typing import Iterator, Tuple, Union
from ...errors import Errors
from ...symbols import NOUN, PRON, PROPN
from ...tokens import Doc, Span

    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    