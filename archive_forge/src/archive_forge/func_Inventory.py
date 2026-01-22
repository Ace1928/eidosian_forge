import boto
import boto.utils
from boto.compat import StringIO
from boto.mashups.iobject import IObject
from boto.pyami.config import Config, BotoConfigPath
from boto.mashups.interactive import interactive_shell
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty
import os
@classmethod
def Inventory(cls):
    """
        Returns a list of Server instances, one for each Server object
        persisted in the db
        """
    l = ServerSet()
    rs = cls.find()
    for server in rs:
        l.append(server)
    return l