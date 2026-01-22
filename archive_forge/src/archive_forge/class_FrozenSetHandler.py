import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class FrozenSetHandler(SetHandler):
    identity = 5
    handles = (frozenset,)