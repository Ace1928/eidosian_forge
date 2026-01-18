import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
Generate a Keystone V3 token based on auth request.