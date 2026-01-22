import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
def assertTimestampsEqual(self, expected, actual):
    exp_time = timeutils.parse_isotime(expected)
    actual_time = timeutils.parse_isotime(actual)
    return self.assertCloseEnoughForGovernmentWork(exp_time, actual_time, delta=1e-05)