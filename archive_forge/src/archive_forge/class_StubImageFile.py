from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
class StubImageFile(ImageFile):
    """
    Base class for stub image loaders.

    A stub loader is an image loader that can identify files of a
    certain format, but relies on external code to load the file.
    """

    def _open(self):
        msg = 'StubImageFile subclass must implement _open'
        raise NotImplementedError(msg)

    def load(self):
        loader = self._load()
        if loader is None:
            msg = f'cannot find loader for this {self.format} file'
            raise OSError(msg)
        image = loader.load(self)
        assert image is not None
        self.__class__ = image.__class__
        self.__dict__ = image.__dict__
        return image.load()

    def _load(self):
        """(Hook) Find actual image loader."""
        msg = 'StubImageFile subclass must implement _load'
        raise NotImplementedError(msg)