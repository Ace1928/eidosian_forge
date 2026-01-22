import logging
import time
from datetime import datetime
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty
from boto.sdb.db.property import DateTimeProperty, FloatProperty, ReferenceProperty
from boto.sdb.db.property import PasswordProperty, ListProperty, MapProperty
from boto.exception import SDBPersistenceError
class TestAutoNow(Model):
    create_date = DateTimeProperty(auto_now_add=True)
    modified_date = DateTimeProperty(auto_now=True)