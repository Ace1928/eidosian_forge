import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
def _set_direct(self, obj, value):
    if not self.use_method:
        setattr(obj, self.slot_name, value)