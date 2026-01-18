import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_provided_replica_set_name(self):
    self.arguments['use_replica'] = True
    self.arguments['replicaset_name'] = 'my_replica'
    dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    self.assertTrue(True)