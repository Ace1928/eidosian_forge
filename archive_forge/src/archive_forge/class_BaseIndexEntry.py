from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
class BaseIndexEntry(BaseIndexEntryHelper):
    """Small brother of an index entry which can be created to describe changes
    done to the index in which case plenty of additional information is not required.

    As the first 4 data members match exactly to the IndexEntry type, methods
    expecting a BaseIndexEntry can also handle full IndexEntries even if they
    use numeric indices for performance reasons.
    """

    def __new__(cls, inp_tuple: Union[Tuple[int, bytes, int, PathLike], Tuple[int, bytes, int, PathLike, bytes, bytes, int, int, int, int, int]]) -> 'BaseIndexEntry':
        """Override __new__ to allow construction from a tuple for backwards compatibility"""
        return super().__new__(cls, *inp_tuple)

    def __str__(self) -> str:
        return '%o %s %i\t%s' % (self.mode, self.hexsha, self.stage, self.path)

    def __repr__(self) -> str:
        return '(%o, %s, %i, %s)' % (self.mode, self.hexsha, self.stage, self.path)

    @property
    def hexsha(self) -> str:
        """hex version of our sha"""
        return b2a_hex(self.binsha).decode('ascii')

    @property
    def stage(self) -> int:
        """Stage of the entry, either:

            * 0 = default stage
            * 1 = stage before a merge or common ancestor entry in case of a 3 way merge
            * 2 = stage of entries from the 'left' side of the merge
            * 3 = stage of entries from the right side of the merge

        :note: For more information, see http://www.kernel.org/pub/software/scm/git/docs/git-read-tree.html
        """
        return (self.flags & CE_STAGEMASK) >> CE_STAGESHIFT

    @classmethod
    def from_blob(cls, blob: Blob, stage: int=0) -> 'BaseIndexEntry':
        """:return: Fully equipped BaseIndexEntry at the given stage"""
        return cls((blob.mode, blob.binsha, stage << CE_STAGESHIFT, blob.path))

    def to_blob(self, repo: 'Repo') -> Blob:
        """:return: Blob using the information of this index entry"""
        return Blob(repo, self.binsha, self.mode, self.path)