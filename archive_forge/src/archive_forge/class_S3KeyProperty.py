import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class S3KeyProperty(Property):
    data_type = boto.s3.key.Key
    type_name = 'S3Key'
    validate_regex = '^s3:\\/\\/([^\\/]*)\\/(.*)$'

    def __init__(self, verbose_name=None, name=None, default=None, required=False, validator=None, choices=None, unique=False):
        super(S3KeyProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)

    def validate(self, value):
        value = super(S3KeyProperty, self).validate(value)
        if value == self.default_value() or value == str(self.default_value()):
            return self.default_value()
        if isinstance(value, self.data_type):
            return
        match = re.match(self.validate_regex, value)
        if match:
            return
        raise TypeError('Validation Error, expecting %s, got %s' % (self.data_type, type(value)))

    def __get__(self, obj, objtype):
        value = super(S3KeyProperty, self).__get__(obj, objtype)
        if value:
            if isinstance(value, self.data_type):
                return value
            match = re.match(self.validate_regex, value)
            if match:
                s3 = obj._manager.get_s3_connection()
                bucket = s3.get_bucket(match.group(1), validate=False)
                k = bucket.get_key(match.group(2))
                if not k:
                    k = bucket.new_key(match.group(2))
                    k.set_contents_from_string('')
                return k
        else:
            return value

    def get_value_for_datastore(self, model_instance):
        value = super(S3KeyProperty, self).get_value_for_datastore(model_instance)
        if value:
            return 's3://%s/%s' % (value.bucket.name, value.name)
        else:
            return None