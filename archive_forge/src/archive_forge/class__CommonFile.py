from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
class _CommonFile(object):
    """common framework for HtpasswdFile & HtdigestFile"""
    encoding = None
    return_unicode = None
    _path = None
    _mtime = None
    autosave = False
    _records = None
    _source = None

    @classmethod
    def from_string(cls, data, **kwds):
        """create new object from raw string.

        :type data: unicode or bytes
        :arg data:
            database to load, as single string.

        :param \\*\\*kwds:
            all other keywords are the same as in the class constructor
        """
        if 'path' in kwds:
            raise TypeError("'path' not accepted by from_string()")
        self = cls(**kwds)
        self.load_string(data)
        return self

    @classmethod
    def from_path(cls, path, **kwds):
        """create new object from file, without binding object to file.

        :type path: str
        :arg path:
            local filepath to load from

        :param \\*\\*kwds:
            all other keywords are the same as in the class constructor
        """
        self = cls(**kwds)
        self.load(path)
        return self

    def __init__(self, path=None, new=False, autoload=True, autosave=False, encoding='utf-8', return_unicode=PY3):
        if not encoding:
            warn('``encoding=None`` is deprecated as of Passlib 1.6, and will cause a ValueError in Passlib 1.8, use ``return_unicode=False`` instead.', DeprecationWarning, stacklevel=2)
            encoding = 'utf-8'
            return_unicode = False
        elif not is_ascii_codec(encoding):
            raise ValueError('encoding must be 7-bit ascii compatible')
        self.encoding = encoding
        self.return_unicode = return_unicode
        self.autosave = autosave
        self._path = path
        self._mtime = 0
        if not autoload:
            warn('``autoload=False`` is deprecated as of Passlib 1.6, and will be removed in Passlib 1.8, use ``new=True`` instead', DeprecationWarning, stacklevel=2)
            new = True
        if path and (not new):
            self.load()
        else:
            self._records = {}
            self._source = []

    def __repr__(self):
        tail = ''
        if self.autosave:
            tail += ' autosave=True'
        if self._path:
            tail += ' path=%r' % self._path
        if self.encoding != 'utf-8':
            tail += ' encoding=%r' % self.encoding
        return '<%s 0x%0x%s>' % (self.__class__.__name__, id(self), tail)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if value != self._path:
            self._mtime = 0
        self._path = value

    @property
    def mtime(self):
        """modify time when last loaded (if bound to a local file)"""
        return self._mtime

    def load_if_changed(self):
        """Reload from ``self.path`` only if file has changed since last load"""
        if not self._path:
            raise RuntimeError('%r is not bound to a local file' % self)
        if self._mtime and self._mtime == os.path.getmtime(self._path):
            return False
        self.load()
        return True

    def load(self, path=None, force=True):
        """Load state from local file.
        If no path is specified, attempts to load from ``self.path``.

        :type path: str
        :arg path: local file to load from

        :type force: bool
        :param force:
            if ``force=False``, only load from ``self.path`` if file
            has changed since last load.

            .. deprecated:: 1.6
                This keyword will be removed in Passlib 1.8;
                Applications should use :meth:`load_if_changed` instead.
        """
        if path is not None:
            with open(path, 'rb') as fh:
                self._mtime = 0
                self._load_lines(fh)
        elif not force:
            warn('%(name)s.load(force=False) is deprecated as of Passlib 1.6,and will be removed in Passlib 1.8; use %(name)s.load_if_changed() instead.' % dict(name=self.__class__.__name__), DeprecationWarning, stacklevel=2)
            return self.load_if_changed()
        elif self._path:
            with open(self._path, 'rb') as fh:
                self._mtime = os.path.getmtime(self._path)
                self._load_lines(fh)
        else:
            raise RuntimeError('%s().path is not set, an explicit path is required' % self.__class__.__name__)
        return True

    def load_string(self, data):
        """Load state from unicode or bytes string, replacing current state"""
        data = to_bytes(data, self.encoding, 'data')
        self._mtime = 0
        self._load_lines(BytesIO(data))

    def _load_lines(self, lines):
        """load from sequence of lists"""
        parse = self._parse_record
        records = {}
        source = []
        skipped = b''
        for idx, line in enumerate(lines):
            tmp = line.lstrip()
            if not tmp or tmp.startswith(_BHASH):
                skipped += line
                continue
            key, value = parse(line, idx + 1)
            if key in records:
                log.warning('username occurs multiple times in source file: %r' % key)
                skipped += line
                continue
            if skipped:
                source.append((_SKIPPED, skipped))
                skipped = b''
            records[key] = value
            source.append((_RECORD, key))
        if skipped.rstrip():
            source.append((_SKIPPED, skipped))
        self._records = records
        self._source = source

    def _parse_record(self, record, lineno):
        """parse line of file into (key, value) pair"""
        raise NotImplementedError('should be implemented in subclass')

    def _set_record(self, key, value):
        """
        helper for setting record which takes care of inserting source line if needed;

        :returns:
            bool if key already present
        """
        records = self._records
        existing = key in records
        records[key] = value
        if not existing:
            self._source.append((_RECORD, key))
        return existing

    def _autosave(self):
        """subclass helper to call save() after any changes"""
        if self.autosave and self._path:
            self.save()

    def save(self, path=None):
        """Save current state to file.
        If no path is specified, attempts to save to ``self.path``.
        """
        if path is not None:
            with open(path, 'wb') as fh:
                fh.writelines(self._iter_lines())
        elif self._path:
            self.save(self._path)
            self._mtime = os.path.getmtime(self._path)
        else:
            raise RuntimeError('%s().path is not set, cannot autosave' % self.__class__.__name__)

    def to_string(self):
        """Export current state as a string of bytes"""
        return join_bytes(self._iter_lines())

    def _iter_lines(self):
        """iterator yielding lines of database"""
        records = self._records
        if __debug__:
            pending = set(records)
        for action, content in self._source:
            if action == _SKIPPED:
                yield content
            else:
                assert action == _RECORD
                if content not in records:
                    continue
                yield self._render_record(content, records[content])
                if __debug__:
                    pending.remove(content)
        if __debug__:
            assert not pending, 'failed to write all records: missing=%r' % (pending,)

    def _render_record(self, key, value):
        """given key/value pair, encode as line of file"""
        raise NotImplementedError('should be implemented in subclass')

    def _encode_user(self, user):
        """user-specific wrapper for _encode_field()"""
        return self._encode_field(user, 'user')

    def _encode_realm(self, realm):
        """realm-specific wrapper for _encode_field()"""
        return self._encode_field(realm, 'realm')

    def _encode_field(self, value, param='field'):
        """convert field to internal representation.

        internal representation is always bytes. byte strings are left as-is,
        unicode strings encoding using file's default encoding (or ``utf-8``
        if no encoding has been specified).

        :raises UnicodeEncodeError:
            if unicode value cannot be encoded using default encoding.

        :raises ValueError:
            if resulting byte string contains a forbidden character,
            or is too long (>255 bytes).

        :returns:
            encoded identifer as bytes
        """
        if isinstance(value, unicode):
            value = value.encode(self.encoding)
        elif not isinstance(value, bytes):
            raise ExpectedStringError(value, param)
        if len(value) > 255:
            raise ValueError('%s must be at most 255 characters: %r' % (param, value))
        if any((c in _INVALID_FIELD_CHARS for c in value)):
            raise ValueError('%s contains invalid characters: %r' % (param, value))
        return value

    def _decode_field(self, value):
        """decode field from internal representation to format
        returns by users() method, etc.

        :raises UnicodeDecodeError:
            if unicode value cannot be decoded using default encoding.
            (usually indicates wrong encoding set for file).

        :returns:
            field as unicode or bytes, as appropriate.
        """
        assert isinstance(value, bytes), 'expected value to be bytes'
        if self.return_unicode:
            return value.decode(self.encoding)
        else:
            return value