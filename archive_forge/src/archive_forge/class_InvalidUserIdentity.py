import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
class InvalidUserIdentity(Exception):
    """User identity is not of the format 'user <email>'."""

    def __init__(self, identity) -> None:
        self.identity = identity