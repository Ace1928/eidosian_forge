from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
class SQLiteSignatureStore(SignatureStore, LoggingConfigurable):
    """Store signatures in an SQLite database."""
    cache_size = Integer(65535, help='The number of notebook signatures to cache.\n        When the number of signatures exceeds this value,\n        the oldest 25% of signatures will be culled.\n        ').tag(config=True)

    def __init__(self, db_file, **kwargs):
        """Initialize a sql signature store."""
        super().__init__(**kwargs)
        self.db_file = db_file
        self.db = self._connect_db(db_file)

    def close(self):
        """Close the db."""
        if self.db is not None:
            self.db.close()

    def _connect_db(self, db_file):
        kwargs: dict[str, t.Any] = {'detect_types': sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES}
        db = None
        try:
            db = sqlite3.connect(db_file, **kwargs)
            self.init_db(db)
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            if db_file != ':memory:':
                old_db_location = db_file + '.bak'
                if db is not None:
                    db.close()
                self.log.warning('The signatures database cannot be opened; maybe it is corrupted or encrypted. You may need to rerun your notebooks to ensure that they are trusted to run Javascript. The old signatures database has been renamed to %s and a new one has been created.', old_db_location)
                try:
                    Path(db_file).rename(old_db_location)
                    db = sqlite3.connect(db_file, **kwargs)
                    self.init_db(db)
                except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError):
                    if db is not None:
                        db.close()
                    self.log.warning('Failed committing signatures database to disk. You may need to move the database file to a non-networked file system, using config option `NotebookNotary.db_file`. Using in-memory signatures database for the remainder of this session.')
                    self.db_file = ':memory:'
                    db = sqlite3.connect(':memory:', **kwargs)
                    self.init_db(db)
            else:
                raise
        return db

    def init_db(self, db):
        """Initialize the db."""
        db.execute('\n            CREATE TABLE IF NOT EXISTS nbsignatures\n            (\n                id integer PRIMARY KEY AUTOINCREMENT,\n                algorithm text,\n                signature text,\n                path text,\n                last_seen timestamp\n            )')
        db.execute('\n            CREATE INDEX IF NOT EXISTS algosig ON nbsignatures(algorithm, signature)\n            ')
        db.commit()

    def store_signature(self, digest, algorithm):
        """Store a signature in the db."""
        if self.db is None:
            return
        if not self.check_signature(digest, algorithm):
            self.db.execute('\n                INSERT INTO nbsignatures (algorithm, signature, last_seen)\n                VALUES (?, ?, ?)\n                ', (algorithm, digest, datetime.now(tz=timezone.utc)))
        else:
            self.db.execute('UPDATE nbsignatures SET last_seen = ? WHERE\n                algorithm = ? AND\n                signature = ?;\n                ', (datetime.now(tz=timezone.utc), algorithm, digest))
        self.db.commit()
        n, = self.db.execute('SELECT Count(*) FROM nbsignatures').fetchone()
        if n > self.cache_size:
            self.cull_db()

    def check_signature(self, digest, algorithm):
        """Check a signature against the db."""
        if self.db is None:
            return False
        r = self.db.execute('SELECT id FROM nbsignatures WHERE\n            algorithm = ? AND\n            signature = ?;\n            ', (algorithm, digest)).fetchone()
        if r is None:
            return False
        self.db.execute('UPDATE nbsignatures SET last_seen = ? WHERE\n            algorithm = ? AND\n            signature = ?;\n            ', (datetime.now(tz=timezone.utc), algorithm, digest))
        self.db.commit()
        return True

    def remove_signature(self, digest, algorithm):
        """Remove a signature from the db."""
        self.db.execute('DELETE FROM nbsignatures WHERE\n                algorithm = ? AND\n                signature = ?;\n            ', (algorithm, digest))
        self.db.commit()

    def cull_db(self):
        """Cull oldest 25% of the trusted signatures when the size limit is reached"""
        self.db.execute('DELETE FROM nbsignatures WHERE id IN (\n            SELECT id FROM nbsignatures ORDER BY last_seen DESC LIMIT -1 OFFSET ?\n        );\n        ', (max(int(0.75 * self.cache_size), 1),))