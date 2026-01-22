import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
class TransportStore(Store):
    """A TransportStore is a Store superclass for Stores that use Transports."""

    def add(self, f, fileid, suffix=None):
        """Add contents of a file into the store.

        f -- A file-like object
        """
        mutter('add store entry %r', fileid)
        names = self._id_to_names(fileid, suffix)
        if self._transport.has_any(names):
            raise BzrError('store %r already contains id %r' % (self._transport.base, fileid))
        self._add(names[0], f)

    def _add(self, relpath, f):
        """Actually add the file to the given location.
        This should be overridden by children.
        """
        raise NotImplementedError('children need to implement this function.')

    def _check_fileid(self, fileid):
        if not isinstance(fileid, bytes):
            raise TypeError('Fileids should be bytestrings: {} {!r}'.format(type(fileid), fileid))
        if b'\\' in fileid or b'/' in fileid:
            raise ValueError('invalid store id %r' % fileid)

    def _id_to_names(self, fileid, suffix):
        """Return the names in the expected order"""
        if suffix is not None:
            fn = self._relpath(fileid, [suffix])
        else:
            fn = self._relpath(fileid)
        fn_gz = fn + '.gz'
        if self._compressed:
            return (fn_gz, fn)
        else:
            return (fn, fn_gz)

    def has_id(self, fileid, suffix=None):
        """See Store.has_id."""
        return self._transport.has_any(self._id_to_names(fileid, suffix))

    def _get_name(self, fileid, suffix=None):
        """A special check, which returns the name of an existing file.

        This is similar in spirit to 'has_id', but it is designed
        to return information about which file the store has.
        """
        for name in self._id_to_names(fileid, suffix=suffix):
            if self._transport.has(name):
                return name
        return None

    def _get(self, filename):
        """Return an vanilla file stream for clients to read from.

        This is the body of a template method on 'get', and should be
        implemented by subclasses.
        """
        raise NotImplementedError

    def get(self, fileid, suffix=None):
        """See Store.get()."""
        names = self._id_to_names(fileid, suffix)
        for name in names:
            try:
                return self._get(name)
            except _mod_transport.NoSuchFile:
                pass
        raise KeyError(fileid)

    def __init__(self, a_transport, prefixed=False, compressed=False, dir_mode=None, file_mode=None, escaped=False):
        super().__init__()
        self._transport = a_transport
        self._prefixed = prefixed
        self._compressed = compressed
        self._suffixes = set()
        self._escaped = escaped
        self._dir_mode = dir_mode
        self._file_mode = file_mode
        if escaped and prefixed:
            self._mapper = versionedfile.HashEscapedPrefixMapper()
        elif not escaped and prefixed:
            self._mapper = versionedfile.HashPrefixMapper()
        elif self._escaped:
            raise ValueError('%r: escaped unprefixed stores are not permitted.' % (self,))
        else:
            self._mapper = versionedfile.PrefixMapper()

    def _iter_files_recursive(self):
        """Iterate through the files in the transport."""
        yield from self._transport.iter_files_recursive()

    def __iter__(self):
        for relpath in self._iter_files_recursive():
            name = os.path.basename(relpath)
            if name.endswith('.gz'):
                name = name[:-3]
            skip = False
            for count in range(len(self._suffixes)):
                for suffix in self._suffixes:
                    if name.endswith('.' + suffix):
                        skip = True
            if not skip:
                yield self._mapper.unmap(name)[0]

    def __len__(self):
        return len(list(self.__iter__()))

    def _relpath(self, fileid, suffixes=None):
        self._check_fileid(fileid)
        if suffixes:
            for suffix in suffixes:
                if suffix not in self._suffixes:
                    raise ValueError('Unregistered suffix %r' % suffix)
                self._check_fileid(suffix.encode('utf-8'))
        else:
            suffixes = []
        path = self._mapper.map((fileid,))
        full_path = '.'.join([path] + suffixes)
        return full_path

    def __repr__(self):
        if self._transport is None:
            return '%s(None)' % self.__class__.__name__
        else:
            return '{}({!r})'.format(self.__class__.__name__, self._transport.base)
    __str__ = __repr__

    def listable(self):
        """Return True if this store is able to be listed."""
        return self._transport.listable()

    def register_suffix(self, suffix):
        """Register a suffix as being expected in this store."""
        self._check_fileid(suffix.encode('utf-8'))
        if suffix == 'gz':
            raise ValueError('You cannot register the "gz" suffix.')
        self._suffixes.add(suffix)

    def total_size(self):
        """Return (count, bytes)

        This is the (compressed) size stored on disk, not the size of
        the content."""
        total = 0
        count = 0
        for relpath in self._transport.iter_files_recursive():
            count += 1
            total += self._transport.stat(relpath).st_size
        return (count, total)