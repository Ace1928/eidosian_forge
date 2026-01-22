import difflib
from bisect import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple
class MaxRecursionDepth(Exception):

    def __init__(self) -> None:
        super().__init__('max recursion depth reached')