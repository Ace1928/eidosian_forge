import os
import contextlib
import itertools
import collections.abc
from abc import ABC, abstractmethod
class _SQLiteManySeqFilesDict(_IndexedSeqFileDict):
    """Read only dictionary interface to many sequential record files.

    This code is used in both Bio.SeqIO for indexing as SeqRecord
    objects, and in Bio.SearchIO for indexing QueryResult objects.

    Keeps the keys, file-numbers and offsets in an SQLite database. To access
    a record by key, reads from the offset in the appropriate file and then
    parses the record into an object.

    There are OS limits on the number of files that can be open at once,
    so a pool are kept. If a record is required from a closed file, then
    one of the open handles is closed first.
    """

    def __init__(self, index_filename, filenames, proxy_factory, fmt, key_function, repr, max_open=10):
        """Initialize the class."""
        if sqlite3 is None:
            from Bio import MissingPythonDependencyError
            raise MissingPythonDependencyError('Python was compiled without the sqlite3 module')
        if filenames is not None:
            filenames = list(filenames)
        self._index_filename = index_filename
        self._filenames = filenames
        self._format = fmt
        self._key_function = key_function
        self._proxy_factory = proxy_factory
        self._repr = repr
        self._max_open = max_open
        self._proxies = {}
        self._relative_path = os.path.abspath(os.path.dirname(index_filename))
        if os.path.isfile(index_filename):
            self._load_index()
        else:
            self._build_index()

    def _load_index(self):
        """Call from __init__ to re-use an existing index (PRIVATE)."""
        index_filename = self._index_filename
        relative_path = self._relative_path
        filenames = self._filenames
        fmt = self._format
        proxy_factory = self._proxy_factory
        con = sqlite3.dbapi2.connect(index_filename, check_same_thread=False)
        self._con = con
        try:
            count, = con.execute('SELECT value FROM meta_data WHERE key=?;', ('count',)).fetchone()
            self._length = int(count)
            if self._length == -1:
                con.close()
                raise ValueError('Unfinished/partial database') from None
            count, = con.execute('SELECT MAX(_ROWID_) FROM offset_data;').fetchone()
            if self._length != int(count):
                con.close()
                raise ValueError('Corrupt database? %i entries not %i' % (int(count), self._length)) from None
            self._format, = con.execute('SELECT value FROM meta_data WHERE key=?;', ('format',)).fetchone()
            if fmt and fmt != self._format:
                con.close()
                raise ValueError(f'Index file says format {self._format}, not {fmt}') from None
            try:
                filenames_relative_to_index, = con.execute('SELECT value FROM meta_data WHERE key=?;', ('filenames_relative_to_index',)).fetchone()
                filenames_relative_to_index = filenames_relative_to_index.upper() == 'TRUE'
            except TypeError:
                filenames_relative_to_index = False
            self._filenames = [row[0] for row in con.execute('SELECT name FROM file_data ORDER BY file_number;').fetchall()]
            if filenames_relative_to_index:
                relative_path = os.path.abspath(os.path.dirname(index_filename))
                tmp = []
                for f in self._filenames:
                    if os.path.isabs(f):
                        tmp.append(f)
                    else:
                        tmp.append(os.path.join(relative_path, f.replace('/', os.path.sep)))
                self._filenames = tmp
                del tmp
            if filenames and len(filenames) != len(self._filenames):
                con.close()
                raise ValueError('Index file says %i files, not %i' % (len(self._filenames), len(filenames))) from None
            if filenames and filenames != self._filenames:
                for old, new in zip(self._filenames, filenames):
                    if os.path.abspath(old) != os.path.abspath(new):
                        con.close()
                        if filenames_relative_to_index:
                            raise ValueError('Index file has different filenames, e.g. %r != %r' % (os.path.abspath(old), os.path.abspath(new))) from None
                        else:
                            raise ValueError('Index file has different filenames [This is an old index where any relative paths were relative to the original working directory]. e.g. %r != %r' % (os.path.abspath(old), os.path.abspath(new))) from None
        except sqlite3.OperationalError as err:
            con.close()
            raise ValueError(f'Not a Biopython index database? {err}') from None
        if not proxy_factory(self._format):
            con.close()
            raise ValueError(f"Unsupported format '{self._format}'")

    def _build_index(self):
        """Call from __init__ to create a new index (PRIVATE)."""
        index_filename = self._index_filename
        relative_path = self._relative_path
        filenames = self._filenames
        fmt = self._format
        key_function = self._key_function
        proxy_factory = self._proxy_factory
        max_open = self._max_open
        random_access_proxies = self._proxies
        if not fmt or not filenames:
            raise ValueError(f'Filenames to index and format required to build {index_filename!r}')
        if not proxy_factory(fmt):
            raise ValueError(f"Unsupported format '{fmt}'")
        con = sqlite3.dbapi2.connect(index_filename)
        self._con = con
        con.execute('PRAGMA synchronous=OFF')
        con.execute('PRAGMA locking_mode=EXCLUSIVE')
        con.execute('CREATE TABLE meta_data (key TEXT, value TEXT);')
        con.execute('INSERT INTO meta_data (key, value) VALUES (?,?);', ('count', -1))
        con.execute('INSERT INTO meta_data (key, value) VALUES (?,?);', ('format', fmt))
        con.execute('INSERT INTO meta_data (key, value) VALUES (?,?);', ('filenames_relative_to_index', 'True'))
        con.execute('CREATE TABLE file_data (file_number INTEGER, name TEXT);')
        con.execute('CREATE TABLE offset_data (key TEXT, file_number INTEGER, offset INTEGER, length INTEGER);')
        count = 0
        for file_index, filename in enumerate(filenames):
            f = os.path.abspath(filename)
            if not os.path.isabs(filename) and (not os.path.isabs(index_filename)):
                f = os.path.relpath(filename, relative_path).replace(os.path.sep, '/')
            elif (os.path.dirname(os.path.abspath(filename)) + os.path.sep).startswith(relative_path + os.path.sep):
                f = os.path.relpath(filename, relative_path).replace(os.path.sep, '/')
                assert not f.startswith('../'), f
            con.execute('INSERT INTO file_data (file_number, name) VALUES (?,?);', (file_index, f))
            random_access_proxy = proxy_factory(fmt, filename)
            if key_function:
                offset_iter = ((key_function(key), file_index, offset, length) for key, offset, length in random_access_proxy)
            else:
                offset_iter = ((key, file_index, offset, length) for key, offset, length in random_access_proxy)
            while True:
                batch = list(itertools.islice(offset_iter, 100))
                if not batch:
                    break
                con.executemany('INSERT INTO offset_data (key,file_number,offset,length) VALUES (?,?,?,?);', batch)
                con.commit()
                count += len(batch)
            if len(random_access_proxies) < max_open:
                random_access_proxies[file_index] = random_access_proxy
            else:
                random_access_proxy._handle.close()
        self._length = count
        try:
            con.execute('CREATE UNIQUE INDEX IF NOT EXISTS key_index ON offset_data(key);')
        except sqlite3.IntegrityError as err:
            self._proxies = random_access_proxies
            self.close()
            con.close()
            raise ValueError(f'Duplicate key? {err}') from None
        con.execute('PRAGMA locking_mode=NORMAL')
        con.execute('UPDATE meta_data SET value = ? WHERE key = ?;', (count, 'count'))
        con.commit()

    def __repr__(self):
        return self._repr

    def __contains__(self, key):
        return bool(self._con.execute('SELECT key FROM offset_data WHERE key=?;', (key,)).fetchone())

    def __len__(self):
        """Return the number of records indexed."""
        return self._length

    def __iter__(self):
        """Iterate over the keys."""
        for row in self._con.execute('SELECT key FROM offset_data ORDER BY file_number, offset;'):
            yield str(row[0])

    def __getitem__(self, key):
        """Return record for the specified key."""
        row = self._con.execute('SELECT file_number, offset FROM offset_data WHERE key=?;', (key,)).fetchone()
        if not row:
            raise KeyError
        file_number, offset = row
        proxies = self._proxies
        if file_number in proxies:
            record = proxies[file_number].get(offset)
        else:
            if len(proxies) >= self._max_open:
                proxies.popitem()[1]._handle.close()
            proxy = self._proxy_factory(self._format, self._filenames[file_number])
            record = proxy.get(offset)
            proxies[file_number] = proxy
        if self._key_function:
            key2 = self._key_function(record.id)
        else:
            key2 = record.id
        if key != key2:
            raise ValueError(f'Key did not match ({key} vs {key2})')
        return record

    def get_raw(self, key):
        """Return the raw record from the file as a bytes string.

        If the key is not found, a KeyError exception is raised.
        """
        row = self._con.execute('SELECT file_number, offset, length FROM offset_data WHERE key=?;', (key,)).fetchone()
        if not row:
            raise KeyError
        file_number, offset, length = row
        proxies = self._proxies
        if file_number in proxies:
            if length:
                h = proxies[file_number]._handle
                h.seek(offset)
                return h.read(length)
            else:
                return proxies[file_number].get_raw(offset)
        else:
            if len(proxies) >= self._max_open:
                proxies.popitem()[1]._handle.close()
            proxy = self._proxy_factory(self._format, self._filenames[file_number])
            proxies[file_number] = proxy
            if length:
                h = proxy._handle
                h.seek(offset)
                return h.read(length)
            else:
                return proxy.get_raw(offset)

    def close(self):
        """Close any open file handles."""
        proxies = self._proxies
        while proxies:
            proxies.popitem()[1]._handle.close()