import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class CountHandler(object):
    identity = 2
    handles = (itertools.count,)

    @staticmethod
    def serialize(obj):
        obj = str(obj)
        start = obj.find('(') + 1
        end = obj.rfind(')')
        pieces = obj[start:end].split(',')
        if len(pieces) == 1:
            start = int(pieces[0])
            step = 1
        else:
            start = int(pieces[0])
            step = int(pieces[1])
        return msgpack.packb([start, step])

    @staticmethod
    def deserialize(data):
        value = msgpack.unpackb(data)
        start, step = value
        return itertools.count(start, step)