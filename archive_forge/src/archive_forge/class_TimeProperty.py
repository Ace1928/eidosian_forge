import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class TimeProperty(Property):
    data_type = datetime.time
    type_name = 'Time'

    def __init__(self, verbose_name=None, name=None, default=None, required=False, validator=None, choices=None, unique=False):
        super(TimeProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)

    def validate(self, value):
        value = super(TimeProperty, self).validate(value)
        if value is None:
            return
        if not isinstance(value, self.data_type):
            raise TypeError('Validation Error, expecting %s, got %s' % (self.data_type, type(value)))