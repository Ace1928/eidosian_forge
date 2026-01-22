from git.util import IterableList, join_path
import git.diff as git_diff
from git.util import to_bin_sha
from . import util
from .base import IndexObject, IndexObjUnion
from .blob import Blob
from .submodule.base import Submodule
from .fun import tree_entries_from_data, tree_to_stream
from typing import (
from git.types import PathLike, Literal
class Tree(IndexObject, git_diff.Diffable, util.Traversable, util.Serializable):
    """Tree objects represent an ordered list of Blobs and other Trees.

    ``Tree as a list``::

        Access a specific blob using the
        tree['filename'] notation.

        You may as well access by index
        blob = tree[0]
    """
    type: Literal['tree'] = 'tree'
    __slots__ = ('_cache',)
    commit_id = 14
    blob_id = 8
    symlink_id = 10
    tree_id = 4
    _map_id_to_type: Dict[int, Type[IndexObjUnion]] = {commit_id: Submodule, blob_id: Blob, symlink_id: Blob}

    def __init__(self, repo: 'Repo', binsha: bytes, mode: int=tree_id << 12, path: Union[PathLike, None]=None):
        super().__init__(repo, binsha, mode, path)

    @classmethod
    def _get_intermediate_items(cls, index_object: IndexObjUnion) -> Union[Tuple['Tree', ...], Tuple[()]]:
        if index_object.type == 'tree':
            return tuple(index_object._iter_convert_to_object(index_object._cache))
        return ()

    def _set_cache_(self, attr: str) -> None:
        if attr == '_cache':
            ostream = self.repo.odb.stream(self.binsha)
            self._cache: List[TreeCacheTup] = tree_entries_from_data(ostream.read())
        else:
            super()._set_cache_(attr)

    def _iter_convert_to_object(self, iterable: Iterable[TreeCacheTup]) -> Iterator[IndexObjUnion]:
        """Iterable yields tuples of (binsha, mode, name), which will be converted
        to the respective object representation.
        """
        for binsha, mode, name in iterable:
            path = join_path(self.path, name)
            try:
                yield self._map_id_to_type[mode >> 12](self.repo, binsha, mode, path)
            except KeyError as e:
                raise TypeError("Unknown mode %o found in tree data for path '%s'" % (mode, path)) from e

    def join(self, file: str) -> IndexObjUnion:
        """Find the named object in this tree's contents.

        :return: ``git.Blob`` or ``git.Tree`` or ``git.Submodule``
        :raise KeyError: if given file or tree does not exist in tree
        """
        msg = 'Blob or Tree named %r not found'
        if '/' in file:
            tree = self
            item = self
            tokens = file.split('/')
            for i, token in enumerate(tokens):
                item = tree[token]
                if item.type == 'tree':
                    tree = item
                else:
                    if i != len(tokens) - 1:
                        raise KeyError(msg % file)
                    return item
            if item == self:
                raise KeyError(msg % file)
            return item
        else:
            for info in self._cache:
                if info[2] == file:
                    return self._map_id_to_type[info[1] >> 12](self.repo, info[0], info[1], join_path(self.path, info[2]))
            raise KeyError(msg % file)

    def __truediv__(self, file: str) -> IndexObjUnion:
        """The ``/`` operator is another syntax for joining.

        See :meth:`join` for details.
        """
        return self.join(file)

    @property
    def trees(self) -> List['Tree']:
        """:return: list(Tree, ...) list of trees directly below this tree"""
        return [i for i in self if i.type == 'tree']

    @property
    def blobs(self) -> List[Blob]:
        """:return: list(Blob, ...) list of blobs directly below this tree"""
        return [i for i in self if i.type == 'blob']

    @property
    def cache(self) -> TreeModifier:
        """
        :return: An object allowing to modify the internal cache. This can be used
            to change the tree's contents. When done, make sure you call ``set_done``
            on the tree modifier, or serialization behaviour will be incorrect.
            See :class:`TreeModifier` for more information on how to alter the cache.
        """
        return TreeModifier(self._cache)

    def traverse(self, predicate: Callable[[Union[IndexObjUnion, TraversedTreeTup], int], bool]=lambda i, d: True, prune: Callable[[Union[IndexObjUnion, TraversedTreeTup], int], bool]=lambda i, d: False, depth: int=-1, branch_first: bool=True, visit_once: bool=False, ignore_self: int=1, as_edge: bool=False) -> Union[Iterator[IndexObjUnion], Iterator[TraversedTreeTup]]:
        """For documentation, see util.Traversable._traverse().

        Trees are set to ``visit_once = False`` to gain more performance in the traversal.
        """
        return cast(Union[Iterator[IndexObjUnion], Iterator[TraversedTreeTup]], super()._traverse(predicate, prune, depth, branch_first, visit_once, ignore_self))

    def list_traverse(self, *args: Any, **kwargs: Any) -> IterableList[IndexObjUnion]:
        """
        :return: IterableList with the results of the traversal as produced by
            traverse()
            Tree -> IterableList[Union['Submodule', 'Tree', 'Blob']]
        """
        return super()._list_traverse(*args, **kwargs)

    def __getslice__(self, i: int, j: int) -> List[IndexObjUnion]:
        return list(self._iter_convert_to_object(self._cache[i:j]))

    def __iter__(self) -> Iterator[IndexObjUnion]:
        return self._iter_convert_to_object(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, item: Union[str, int, slice]) -> IndexObjUnion:
        if isinstance(item, int):
            info = self._cache[item]
            return self._map_id_to_type[info[1] >> 12](self.repo, info[0], info[1], join_path(self.path, info[2]))
        if isinstance(item, str):
            return self.join(item)
        raise TypeError('Invalid index type: %r' % item)

    def __contains__(self, item: Union[IndexObjUnion, PathLike]) -> bool:
        if isinstance(item, IndexObject):
            for info in self._cache:
                if item.binsha == info[0]:
                    return True
        else:
            path = self.path
            for info in self._cache:
                if item == join_path(path, info[2]):
                    return True
        return False

    def __reversed__(self) -> Iterator[IndexObjUnion]:
        return reversed(self._iter_convert_to_object(self._cache))

    def _serialize(self, stream: 'BytesIO') -> 'Tree':
        """Serialize this tree into the stream. Assumes sorted tree data.

        .. note:: We will assume our tree data to be in a sorted state. If this is not
            the case, serialization will not generate a correct tree representation as
            these are assumed to be sorted by algorithms.
        """
        tree_to_stream(self._cache, stream.write)
        return self

    def _deserialize(self, stream: 'BytesIO') -> 'Tree':
        self._cache = tree_entries_from_data(stream.read())
        return self