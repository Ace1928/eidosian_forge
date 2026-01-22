import collections
import collections.abc
import datetime
import functools
import io
import ipaddress
import itertools
import json
from unittest import mock
from xmlrpc import client as xmlrpclib
import netaddr
from oslo_i18n import fixture
from oslotest import base as test_base
from oslo_serialization import jsonutils
class IterClass(object):

    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
        self.index = 0

    def __iter__(self):
        return self

    def next(self):
        if self.index == len(self.data):
            raise StopIteration
        self.index = self.index + 1
        return self.data[self.index - 1]
    __next__ = next