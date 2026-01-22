import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
class _VersionsEqual(tt_matchers.MatchesListwise):

    def __init__(self, expected):
        super(_VersionsEqual, self).__init__([tt_matchers.KeysEqual(expected), tt_matchers.KeysEqual(expected['versions']), tt_matchers.HasLength(len(expected['versions']['values'])), tt_matchers.ContainsAll(expected['versions']['values'])])

    def match(self, other):
        return super(_VersionsEqual, self).match([other, other['versions'], other['versions']['values'], other['versions']['values']])