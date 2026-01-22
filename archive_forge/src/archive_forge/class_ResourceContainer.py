import abc
import io
import itertools
from typing import BinaryIO, List
from .abc import Traversable, TraversableResources
class ResourceContainer(Traversable):
    """
    Traversable container for a package's resources via its reader.
    """

    def __init__(self, reader):
        self.reader = reader

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def iterdir(self):
        files = (ResourceHandle(self, name) for name in self.reader.resources)
        dirs = map(ResourceContainer, self.reader.children())
        return itertools.chain(files, dirs)

    def open(self, *args, **kwargs):
        raise IsADirectoryError()

    @staticmethod
    def _flatten(compound_names):
        for name in compound_names:
            yield from name.split('/')

    def joinpath(self, *descendants):
        if not descendants:
            return self
        names = self._flatten(descendants)
        target = next(names)
        return next((traversable for traversable in self.iterdir() if traversable.name == target)).joinpath(*names)