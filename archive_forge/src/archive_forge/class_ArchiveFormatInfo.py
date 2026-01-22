import os
import time
import warnings
from typing import Iterator, cast
from .. import errors, pyutils, registry, trace
class ArchiveFormatInfo:

    def __init__(self, extensions):
        self.extensions = extensions