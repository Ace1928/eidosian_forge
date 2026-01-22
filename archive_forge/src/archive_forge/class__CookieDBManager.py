import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
class _CookieDBManager:
    _db = None
    _cache_dir = _os.path.join(_ad.user_cache_dir(), 'py-yfinance')

    @classmethod
    def get_database(cls):
        if cls._db is None:
            cls._initialise()
        return cls._db

    @classmethod
    def close_db(cls):
        if cls._db is not None:
            try:
                cls._db.close()
            except Exception:
                pass

    @classmethod
    def _initialise(cls, cache_dir=None):
        if cache_dir is not None:
            cls._cache_dir = cache_dir
        if not _os.path.isdir(cls._cache_dir):
            try:
                _os.makedirs(cls._cache_dir)
            except OSError as err:
                raise _CookieCacheException(f"Error creating CookieCache folder: '{cls._cache_dir}' reason: {err}")
        elif not (_os.access(cls._cache_dir, _os.R_OK) and _os.access(cls._cache_dir, _os.W_OK)):
            raise _CookieCacheException(f"Cannot read and write in CookieCache folder: '{cls._cache_dir}'")
        cls._db = _peewee.SqliteDatabase(_os.path.join(cls._cache_dir, 'cookies.db'), pragmas={'journal_mode': 'wal', 'cache_size': -64})

    @classmethod
    def set_location(cls, new_cache_dir):
        if cls._db is not None:
            cls._db.close()
            cls._db = None
        cls._cache_dir = new_cache_dir

    @classmethod
    def get_location(cls):
        return cls._cache_dir