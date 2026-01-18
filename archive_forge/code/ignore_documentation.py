import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
Create a IgnoreFilterManager from a repository.

        Args:
          repo: Repository object
        Returns:
          A `IgnoreFilterManager` object
        