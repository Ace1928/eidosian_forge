import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
class _KndxIndex:
    """Manages knit index files

    The index is kept in memory and read on startup, to enable
    fast lookups of revision information.  The cursor of the index
    file is always pointing to the end, making it easy to append
    entries.

    _cache is a cache for fast mapping from version id to a Index
    object.

    _history is a cache for fast mapping from indexes to version ids.

    The index data format is dictionary compressed when it comes to
    parent references; a index entry may only have parents that with a
    lover index number.  As a result, the index is topological sorted.

    Duplicate entries may be written to the index for a single version id
    if this is done then the latter one completely replaces the former:
    this allows updates to correct version and parent information.
    Note that the two entries may share the delta, and that successive
    annotations and references MUST point to the first entry.

    The index file on disc contains a header, followed by one line per knit
    record. The same revision can be present in an index file more than once.
    The first occurrence gets assigned a sequence number starting from 0.

    The format of a single line is
    REVISION_ID FLAGS BYTE_OFFSET LENGTH( PARENT_ID|PARENT_SEQUENCE_ID)* :

    REVISION_ID is a utf8-encoded revision id
    FLAGS is a comma separated list of flags about the record. Values include
        no-eol, line-delta, fulltext.
    BYTE_OFFSET is the ascii representation of the byte offset in the data file
        that the compressed data starts at.
    LENGTH is the ascii representation of the length of the data file.
    PARENT_ID a utf-8 revision id prefixed by a '.' that is a parent of
        REVISION_ID.
    PARENT_SEQUENCE_ID the ascii representation of the sequence number of a
        revision id already in the knit that is a parent of REVISION_ID.
    The ' :' marker is the end of record marker.

    partial writes:
    when a write is interrupted to the index file, it will result in a line
    that does not end in ' :'. If the ' :' is not present at the end of a line,
    or at the end of the file, then the record that is missing it will be
    ignored by the parser.

    When writing new records to the index file, the data is preceded by '
'
    to ensure that records always start on new lines even if the last write was
    interrupted. As a result its normal for the last line in the index to be
    missing a trailing newline. One can be added with no harmful effects.

    :ivar _kndx_cache: dict from prefix to the old state of KnitIndex objects,
        where prefix is e.g. the (fileid,) for .texts instances or () for
        constant-mapped things like .revisions, and the old state is
        tuple(history_vector, cache_dict).  This is used to prevent having an
        ABI change with the C extension that reads .kndx files.
    """
    HEADER = b'# bzr knit index 8\n'

    def __init__(self, transport, mapper, get_scope, allow_writes, is_locked):
        """Create a _KndxIndex on transport using mapper."""
        self._transport = transport
        self._mapper = mapper
        self._get_scope = get_scope
        self._allow_writes = allow_writes
        self._is_locked = is_locked
        self._reset_cache()
        self.has_graph = True

    def add_records(self, records, random_id=False, missing_compression_parents=False):
        """Add multiple records to the index.

        :param records: a list of tuples:
                         (key, options, access_memo, parents).
        :param random_id: If True the ids being added were randomly generated
            and no check for existence will be performed.
        :param missing_compression_parents: If True the records being added are
            only compressed against texts already in the index (or inside
            records). If False the records all refer to unavailable texts (or
            texts inside records) as compression parents.
        """
        if missing_compression_parents:
            keys = sorted((record[0] for record in records))
            raise errors.RevisionNotPresent(keys, self)
        paths = {}
        for record in records:
            key = record[0]
            prefix = key[:-1]
            path = self._mapper.map(key) + '.kndx'
            path_keys = paths.setdefault(path, (prefix, []))
            path_keys[1].append(record)
        for path in sorted(paths):
            prefix, path_keys = paths[path]
            self._load_prefixes([prefix])
            lines = []
            orig_history = self._kndx_cache[prefix][1][:]
            orig_cache = self._kndx_cache[prefix][0].copy()
            try:
                for key, options, (_, pos, size), parents in path_keys:
                    if not all((isinstance(option, bytes) for option in options)):
                        raise TypeError(options)
                    if parents is None:
                        parents = ()
                    line = b' '.join([b'\n' + key[-1], b','.join(options), b'%d' % pos, b'%d' % size, self._dictionary_compress(parents), b':'])
                    if not isinstance(line, bytes):
                        raise AssertionError('data must be utf8 was %s' % type(line))
                    lines.append(line)
                    self._cache_key(key, options, pos, size, parents)
                if len(orig_history):
                    self._transport.append_bytes(path, b''.join(lines))
                else:
                    self._init_index(path, lines)
            except:
                self._kndx_cache[prefix] = (orig_cache, orig_history)
                raise

    def scan_unvalidated_index(self, graph_index):
        """See _KnitGraphIndex.scan_unvalidated_index."""
        raise NotImplementedError(self.scan_unvalidated_index)

    def get_missing_compression_parents(self):
        """See _KnitGraphIndex.get_missing_compression_parents."""
        raise NotImplementedError(self.get_missing_compression_parents)

    def _cache_key(self, key, options, pos, size, parent_keys):
        """Cache a version record in the history array and index cache.

        This is inlined into _load_data for performance. KEEP IN SYNC.
        (It saves 60ms, 25% of the __init__ overhead on local 4000 record
         indexes).
        """
        prefix = key[:-1]
        version_id = key[-1]
        parents = tuple((parent[-1] for parent in parent_keys))
        for parent in parent_keys:
            if parent[:-1] != prefix:
                raise ValueError('mismatched prefixes for {!r}, {!r}'.format(key, parent_keys))
        cache, history = self._kndx_cache[prefix]
        if version_id not in cache:
            index = len(history)
            history.append(version_id)
        else:
            index = cache[version_id][5]
        cache[version_id] = (version_id, options, pos, size, parents, index)

    def check_header(self, fp):
        line = fp.readline()
        if line == b'':
            raise _mod_transport.NoSuchFile(self)
        if line != self.HEADER:
            raise KnitHeaderError(badline=line, filename=self)

    def _check_read(self):
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)
        if self._get_scope() != self._scope:
            self._reset_cache()

    def _check_write_ok(self):
        """Assert if not writes are permitted."""
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)
        if self._get_scope() != self._scope:
            self._reset_cache()
        if self._mode != 'w':
            raise errors.ReadOnlyObjectDirtiedError(self)

    def get_build_details(self, keys):
        """Get the method, index_memo and compression parent for keys.

        Ghosts are omitted from the result.

        :param keys: An iterable of keys.
        :return: A dict of key:(index_memo, compression_parent, parents,
            record_details).
            index_memo
                opaque structure to pass to read_records to extract the raw
                data
            compression_parent
                Content that this record is built upon, may be None
            parents
                Logical parents of this node
            record_details
                extra information about the content which needs to be passed to
                Factory.parse_record
        """
        parent_map = self.get_parent_map(keys)
        result = {}
        for key in keys:
            if key not in parent_map:
                continue
            method = self.get_method(key)
            if not isinstance(method, str):
                raise TypeError(method)
            parents = parent_map[key]
            if method == 'fulltext':
                compression_parent = None
            else:
                compression_parent = parents[0]
            noeol = b'no-eol' in self.get_options(key)
            index_memo = self.get_position(key)
            result[key] = (index_memo, compression_parent, parents, (method, noeol))
        return result

    def get_method(self, key):
        """Return compression method of specified key."""
        options = self.get_options(key)
        if b'fulltext' in options:
            return 'fulltext'
        elif b'line-delta' in options:
            return 'line-delta'
        else:
            raise KnitIndexUnknownMethod(self, options)

    def get_options(self, key):
        """Return a list representing options.

        e.g. ['foo', 'bar']
        """
        prefix, suffix = self._split_key(key)
        self._load_prefixes([prefix])
        try:
            return self._kndx_cache[prefix][0][suffix][1]
        except KeyError:
            raise RevisionNotPresent(key, self)

    def find_ancestry(self, keys):
        """See CombinedGraphIndex.find_ancestry()"""
        prefixes = {key[:-1] for key in keys}
        self._load_prefixes(prefixes)
        result = {}
        parent_map = {}
        missing_keys = set()
        pending_keys = list(keys)
        while pending_keys:
            key = pending_keys.pop()
            if key in parent_map:
                continue
            prefix = key[:-1]
            try:
                suffix_parents = self._kndx_cache[prefix][0][key[-1]][4]
            except KeyError:
                missing_keys.add(key)
            else:
                parent_keys = tuple([prefix + (suffix,) for suffix in suffix_parents])
                parent_map[key] = parent_keys
                pending_keys.extend([p for p in parent_keys if p not in parent_map])
        return (parent_map, missing_keys)

    def get_parent_map(self, keys):
        """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A mapping from keys to parents. Absent keys are absent from
            the mapping.
        """
        prefixes = {key[:-1] for key in keys}
        self._load_prefixes(prefixes)
        result = {}
        for key in keys:
            prefix = key[:-1]
            try:
                suffix_parents = self._kndx_cache[prefix][0][key[-1]][4]
            except KeyError:
                pass
            else:
                result[key] = tuple((prefix + (suffix,) for suffix in suffix_parents))
        return result

    def get_position(self, key):
        """Return details needed to access the version.

        :return: a tuple (key, data position, size) to hand to the access
            logic to get the record.
        """
        prefix, suffix = self._split_key(key)
        self._load_prefixes([prefix])
        entry = self._kndx_cache[prefix][0][suffix]
        return (key, entry[2], entry[3])
    __contains__ = _mod_index._has_key_from_parent_map

    def _init_index(self, path, extra_lines=[]):
        """Initialize an index."""
        sio = BytesIO()
        sio.write(self.HEADER)
        sio.writelines(extra_lines)
        sio.seek(0)
        self._transport.put_file_non_atomic(path, sio, create_parent_dir=True)

    def keys(self):
        """Get all the keys in the collection.

        The keys are not ordered.
        """
        result = set()
        if isinstance(self._mapper, ConstantMapper):
            prefixes = [()]
        else:
            relpaths = set()
            for quoted_relpath in self._transport.iter_files_recursive():
                path, ext = os.path.splitext(quoted_relpath)
                relpaths.add(path)
            prefixes = [self._mapper.unmap(path) for path in relpaths]
        self._load_prefixes(prefixes)
        for prefix in prefixes:
            for suffix in self._kndx_cache[prefix][1]:
                result.add(prefix + (suffix,))
        return result

    def _load_prefixes(self, prefixes):
        """Load the indices for prefixes."""
        self._check_read()
        for prefix in prefixes:
            if prefix not in self._kndx_cache:
                self._cache = {}
                self._history = []
                self._filename = prefix
                try:
                    path = self._mapper.map(prefix) + '.kndx'
                    with self._transport.get(path) as fp:
                        _load_data(self, fp)
                    self._kndx_cache[prefix] = (self._cache, self._history)
                    del self._cache
                    del self._filename
                    del self._history
                except NoSuchFile:
                    self._kndx_cache[prefix] = ({}, [])
                    if isinstance(self._mapper, ConstantMapper):
                        self._init_index(path)
                    del self._cache
                    del self._filename
                    del self._history
    missing_keys = _mod_index._missing_keys_from_parent_map

    def _partition_keys(self, keys):
        """Turn keys into a dict of prefix:suffix_list."""
        result = {}
        for key in keys:
            prefix_keys = result.setdefault(key[:-1], [])
            prefix_keys.append(key[-1])
        return result

    def _dictionary_compress(self, keys):
        """Dictionary compress keys.

        :param keys: The keys to generate references to.
        :return: A string representation of keys. keys which are present are
            dictionary compressed, and others are emitted as fulltext with a
            '.' prefix.
        """
        if not keys:
            return b''
        result_list = []
        prefix = keys[0][:-1]
        cache = self._kndx_cache[prefix][0]
        for key in keys:
            if key[:-1] != prefix:
                raise ValueError('mismatched prefixes for %r' % keys)
            if key[-1] in cache:
                result_list.append(b'%d' % cache[key[-1]][5])
            else:
                result_list.append(b'.' + key[-1])
        return b' '.join(result_list)

    def _reset_cache(self):
        self._kndx_cache = {}
        self._scope = self._get_scope()
        allow_writes = self._allow_writes()
        if allow_writes:
            self._mode = 'w'
        else:
            self._mode = 'r'

    def _sort_keys_by_io(self, keys, positions):
        """Figure out an optimal order to read the records for the given keys.

        Sort keys, grouped by index and sorted by position.

        :param keys: A list of keys whose records we want to read. This will be
            sorted 'in-place'.
        :param positions: A dict, such as the one returned by
            _get_components_positions()
        :return: None
        """

        def get_sort_key(key):
            index_memo = positions[key][1]
            return (index_memo[0][:-1], index_memo[1])
        return keys.sort(key=get_sort_key)
    _get_total_build_size = _get_total_build_size

    def _split_key(self, key):
        """Split key into a prefix and suffix."""
        if isinstance(key, bytes):
            return (key[:-1], key[-1:])
        return (key[:-1], key[-1])