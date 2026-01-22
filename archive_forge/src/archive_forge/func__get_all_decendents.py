import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def _get_all_decendents(self, cls):
    """Get all decendents for a given class"""
    decendents = {}
    for sc in cls.__sub_classes__:
        decendents[sc.__name__] = sc
        decendents.update(self._get_all_decendents(sc))
    return decendents