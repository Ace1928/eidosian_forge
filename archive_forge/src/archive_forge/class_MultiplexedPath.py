import collections
import operator
import pathlib
import zipfile
from . import abc
from ._itertools import unique_everseen
class MultiplexedPath(abc.Traversable):
    """
    Given a series of Traversable objects, implement a merged
    version of the interface across all objects. Useful for
    namespace packages which may be multihomed at a single
    name.
    """

    def __init__(self, *paths):
        self._paths = list(map(pathlib.Path, remove_duplicates(paths)))
        if not self._paths:
            message = 'MultiplexedPath must contain at least one path'
            raise FileNotFoundError(message)
        if not all((path.is_dir() for path in self._paths)):
            raise NotADirectoryError('MultiplexedPath only supports directories')

    def iterdir(self):
        files = (file for path in self._paths for file in path.iterdir())
        return unique_everseen(files, key=operator.attrgetter('name'))

    def read_bytes(self):
        raise FileNotFoundError(f'{self} is not a file')

    def read_text(self, *args, **kwargs):
        raise FileNotFoundError(f'{self} is not a file')

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def joinpath(self, child):
        for file in self.iterdir():
            if file.name == child:
                return file
        return self._paths[0] / child
    __truediv__ = joinpath

    def open(self, *args, **kwargs):
        raise FileNotFoundError(f'{self} is not a file')

    @property
    def name(self):
        return self._paths[0].name

    def __repr__(self):
        paths = ', '.join((f"'{path}'" for path in self._paths))
        return f'MultiplexedPath({paths})'