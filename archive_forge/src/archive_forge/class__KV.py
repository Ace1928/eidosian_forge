import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
class _KV(_peewee.Model):
    key = _peewee.CharField(primary_key=True)
    value = _peewee.CharField(null=True)

    class Meta:
        database = tz_db_proxy
        without_rowid = True