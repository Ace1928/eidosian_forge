import re
from typing import Any, List, Tuple, Union
from .errors import ParseError
def to_slice(self) -> slice:
    """Return the slice equivalent of this page range."""
    return self._slice