from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class RefLog(List[RefLogEntry], Serializable):
    """A reflog contains RefLogEntrys, each of which defines a certain state
    of the head in question. Custom query methods allow to retrieve log entries
    by date or by other criteria.

    Reflog entries are ordered. The first added entry is first in the list. The last
    entry, i.e. the last change of the head or reference, is last in the list.
    """
    __slots__ = ('_path',)

    def __new__(cls, filepath: Union[PathLike, None]=None) -> 'RefLog':
        inst = super().__new__(cls)
        return inst

    def __init__(self, filepath: Union[PathLike, None]=None):
        """Initialize this instance with an optional filepath, from which we will
        initialize our data. The path is also used to write changes back using
        the write() method."""
        self._path = filepath
        if filepath is not None:
            self._read_from_file()

    def _read_from_file(self) -> None:
        try:
            fmap = file_contents_ro_filepath(self._path, stream=True, allow_mmap=True)
        except OSError:
            return
        try:
            self._deserialize(fmap)
        finally:
            fmap.close()

    @classmethod
    def from_file(cls, filepath: PathLike) -> 'RefLog':
        """
        :return: A new RefLog instance containing all entries from the reflog
            at the given filepath
        :param filepath: Path to reflog
        :raise ValueError: If the file could not be read or was corrupted in some way
        """
        return cls(filepath)

    @classmethod
    def path(cls, ref: 'SymbolicReference') -> str:
        """
        :return: String to absolute path at which the reflog of the given ref
            instance would be found. The path is not guaranteed to point to a valid
            file though.
        :param ref: SymbolicReference instance
        """
        return osp.join(ref.repo.git_dir, 'logs', to_native_path(ref.path))

    @classmethod
    def iter_entries(cls, stream: Union[str, 'BytesIO', mmap]) -> Iterator[RefLogEntry]:
        """
        :return: Iterator yielding RefLogEntry instances, one for each line read
            from the given stream.

        :param stream: File-like object containing the revlog in its native format
            or string instance pointing to a file to read.
        """
        new_entry = RefLogEntry.from_line
        if isinstance(stream, str):
            _stream = file_contents_ro_filepath(stream)
            assert isinstance(_stream, mmap)
        else:
            _stream = stream
        while True:
            line = _stream.readline()
            if not line:
                return
            yield new_entry(line.strip())

    @classmethod
    def entry_at(cls, filepath: PathLike, index: int) -> 'RefLogEntry':
        """
        :return: RefLogEntry at the given index.

        :param filepath: Full path to the index file from which to read the entry.

        :param index: Python list compatible index, i.e. it may be negative to
            specify an entry counted from the end of the list.

        :raise IndexError: If the entry didn't exist.

        .. note:: This method is faster as it only parses the entry at index, skipping
            all other lines. Nonetheless, the whole file has to be read if
            the index is negative.
        """
        with open(filepath, 'rb') as fp:
            if index < 0:
                return RefLogEntry.from_line(fp.readlines()[index].strip())
            for i in range(index + 1):
                line = fp.readline()
                if not line:
                    raise IndexError(f'Index file ended at line {i + 1}, before given index was reached')
            return RefLogEntry.from_line(line.strip())

    def to_file(self, filepath: PathLike) -> None:
        """Write the contents of the reflog instance to a file at the given filepath.

        :param filepath: Path to file, parent directories are assumed to exist.
        """
        lfd = LockedFD(filepath)
        assure_directory_exists(filepath, is_file=True)
        fp = lfd.open(write=True, stream=True)
        try:
            self._serialize(fp)
            lfd.commit()
        except BaseException:
            lfd.rollback()
            raise

    @classmethod
    def append_entry(cls, config_reader: Union[Actor, 'GitConfigParser', 'SectionConstraint', None], filepath: PathLike, oldbinsha: bytes, newbinsha: bytes, message: str, write: bool=True) -> 'RefLogEntry':
        """Append a new log entry to the revlog at filepath.

        :param config_reader: Configuration reader of the repository - used to obtain
            user information. May also be an Actor instance identifying the committer
            directly or None.
        :param filepath: Full path to the log file.
        :param oldbinsha: Binary sha of the previous commit.
        :param newbinsha: Binary sha of the current commit.
        :param message: Message describing the change to the reference.
        :param write: If True, the changes will be written right away. Otherwise the
            change will not be written.

        :return: RefLogEntry objects which was appended to the log.

        :note: As we are append-only, concurrent access is not a problem as we do not
            interfere with readers.
        """
        if len(oldbinsha) != 20 or len(newbinsha) != 20:
            raise ValueError('Shas need to be given in binary format')
        assure_directory_exists(filepath, is_file=True)
        first_line = message.split('\n')[0]
        if isinstance(config_reader, Actor):
            committer = config_reader
        else:
            committer = Actor.committer(config_reader)
        entry = RefLogEntry((bin_to_hex(oldbinsha).decode('ascii'), bin_to_hex(newbinsha).decode('ascii'), committer, (int(_time.time()), _time.altzone), first_line))
        if write:
            lf = LockFile(filepath)
            lf._obtain_lock_or_raise()
            fd = open(filepath, 'ab')
            try:
                fd.write(entry.format().encode(defenc))
            finally:
                fd.close()
                lf._release_lock()
        return entry

    def write(self) -> 'RefLog':
        """Write this instance's data to the file we are originating from.

        :return: self
        """
        if self._path is None:
            raise ValueError('Instance was not initialized with a path, use to_file(...) instead')
        self.to_file(self._path)
        return self

    def _serialize(self, stream: 'BytesIO') -> 'RefLog':
        write = stream.write
        for e in self:
            write(e.format().encode(defenc))
        return self

    def _deserialize(self, stream: 'BytesIO') -> 'RefLog':
        self.extend(self.iter_entries(stream))
        return self