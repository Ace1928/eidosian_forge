import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
class PackContainer(Protocol):

    def add_pack(self) -> Tuple[BytesIO, Callable[[], None], Callable[[], None]]:
        """Add a new pack."""