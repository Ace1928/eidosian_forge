import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def add_son_manipulator(self, manipulator):
    global SON_MANIPULATOR
    SON_MANIPULATOR = manipulator