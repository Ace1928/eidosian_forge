import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
class ColorHandler(object):
    handles = (Color,)
    identity = msgpackutils.HandlerRegistry.non_reserved_extension_range.min_value + 1

    @staticmethod
    def serialize(obj):
        blob = '%s, %s, %s' % (obj.r, obj.g, obj.b)
        blob = blob.encode('ascii')
        return blob

    @staticmethod
    def deserialize(data):
        chunks = [int(c.strip()) for c in data.split(b',')]
        return Color(chunks[0], chunks[1], chunks[2])