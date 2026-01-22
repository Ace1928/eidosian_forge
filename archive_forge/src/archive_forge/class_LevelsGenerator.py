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
class LevelsGenerator(object):

    def __init__(self, levels):
        self._levels = levels

    def iteritems(self):
        if self._levels == 0:
            return iter([])
        else:
            return iter([(0, LevelsGenerator(self._levels - 1))])