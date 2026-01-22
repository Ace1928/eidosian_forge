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
def _build_filter(self, property, name, op, val):
    if name == '__id__':
        name = 'itemName()'
    if name != 'itemName()':
        name = '`%s`' % name
    if val is None:
        if op in ('is', '='):
            return '%(name)s is null' % {'name': name}
        elif op in ('is not', '!='):
            return '%s is not null' % name
        else:
            val = ''
    if property.__class__ == ListProperty:
        if op in ('is', '='):
            op = 'like'
        elif op in ('!=', 'not'):
            op = 'not like'
        if not (op in ['like', 'not like'] and val.startswith('%')):
            val = '%%:%s' % val
    return "%s %s '%s'" % (name, op, val.replace("'", "''"))