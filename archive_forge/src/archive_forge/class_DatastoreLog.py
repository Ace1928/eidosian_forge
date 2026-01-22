import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
class DatastoreLog(base.Resource):
    """A DatastoreLog is a log on the database guest instance."""

    def __repr__(self):
        return '<DatastoreLog: %s>' % self.name