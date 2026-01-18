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
def test_precedence_items_iteritems(self):

    class ItemsIterItemsClass(object):

        def items(self):
            return {'items': 'items'}

        def iteritems(self):
            return {'iteritems': 'iteritems'}
    x = ItemsIterItemsClass()
    p = jsonutils.to_primitive(x)
    self.assertEqual({'iteritems': 'iteritems'}, p)