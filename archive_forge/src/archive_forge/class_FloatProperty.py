import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class FloatProperty(Property):
    data_type = float
    type_name = 'Float'

    def __init__(self, verbose_name=None, name=None, default=0.0, required=False, validator=None, choices=None, unique=False):
        super(FloatProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)

    def validate(self, value):
        value = float(value)
        value = super(FloatProperty, self).validate(value)
        return value

    def empty(self, value):
        return value is None