import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def _dumps_loads(obj):
    obj = msgpackutils.dumps(obj)
    return msgpackutils.loads(obj)