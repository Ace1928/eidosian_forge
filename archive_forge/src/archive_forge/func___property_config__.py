import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
def __property_config__(self, model_class, property_name):
    super(ReferenceProperty, self).__property_config__(model_class, property_name)
    if self.collection_name is None:
        self.collection_name = '%s_%s_set' % (model_class.__name__.lower(), self.name)
    if hasattr(self.reference_class, self.collection_name):
        raise ValueError('duplicate property: %s' % self.collection_name)
    setattr(self.reference_class, self.collection_name, _ReverseReferenceProperty(model_class, property_name, self.collection_name))