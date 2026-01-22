import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
def append_pattern(self, pattern: bytes) -> None:
    """Add a pattern to the set."""
    self._patterns.append(Pattern(pattern, self._ignorecase))