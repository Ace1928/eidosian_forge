import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class DateProperty(Property):
    data_type = datetime.date
    type_name = 'Date'

    def __init__(self, verbose_name=None, auto_now=False, auto_now_add=False, name=None, default=None, required=False, validator=None, choices=None, unique=False):
        super(DateProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add

    def default_value(self):
        if self.auto_now or self.auto_now_add:
            return self.now()
        return super(DateProperty, self).default_value()

    def validate(self, value):
        value = super(DateProperty, self).validate(value)
        if value is None:
            return
        if not isinstance(value, self.data_type):
            raise TypeError('Validation Error, expecting %s, got %s' % (self.data_type, type(value)))

    def get_value_for_datastore(self, model_instance):
        if self.auto_now:
            setattr(model_instance, self.name, self.now())
        val = super(DateProperty, self).get_value_for_datastore(model_instance)
        if isinstance(val, datetime.datetime):
            val = val.date()
        return val

    def now(self):
        return datetime.date.today()