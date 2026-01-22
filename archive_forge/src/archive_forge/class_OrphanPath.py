from contextlib import suppress
from io import TextIOWrapper
from . import abc
class OrphanPath(abc.Traversable):
    """
        Orphan path, not tied to a module spec or resource reader.
        Can't be read and doesn't expose any meaningful children.
        """

    def __init__(self, *path_parts):
        if len(path_parts) < 1:
            raise ValueError('Need at least one path part to construct a path')
        self._path = path_parts

    def iterdir(self):
        return iter(())

    def is_file(self):
        return False
    is_dir = is_file

    def joinpath(self, other):
        return CompatibilityFiles.OrphanPath(*self._path, other)

    @property
    def name(self):
        return self._path[-1]

    def open(self, mode='r', *args, **kwargs):
        raise FileNotFoundError("Can't open orphan path")