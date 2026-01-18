import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
def validate_string(value):
    if value is None:
        return
    elif isinstance(value, six.string_types):
        if len(value) > 1024:
            raise ValueError('Length of value greater than maxlength')
    else:
        raise TypeError('Expecting String, got %s' % type(value))