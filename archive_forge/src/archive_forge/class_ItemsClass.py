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
class ItemsClass(object):

    def __init__(self):
        self.data = dict(a=1, b=2, c=3)

    def items(self):
        return self.data.items()