import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class StringProperty(Property):
    type_name = 'String'

    def __init__(self, verbose_name=None, name=None, default='', required=False, validator=validate_string, choices=None, unique=False):
        super(StringProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)