import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def add_star_prefix(self) -> None:
    """
        Append '*/' to the path to keep the context constrained
        to a single parent.
        """
    self.path += '*/'