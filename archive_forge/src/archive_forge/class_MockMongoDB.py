import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
class MockMongoDB(object):

    def __init__(self, dbname):
        self._dbname = dbname

    def authenticate(self, username, password):
        pass

    def add_son_manipulator(self, manipulator):
        global SON_MANIPULATOR
        SON_MANIPULATOR = manipulator

    def __getattr__(self, name):
        if name == 'authenticate':
            return self.authenticate
        elif name == 'name':
            return self._dbname
        elif name == 'add_son_manipulator':
            return self.add_son_manipulator
        else:
            return get_collection(self._dbname, name)

    def __getitem__(self, name):
        return get_collection(self._dbname, name)