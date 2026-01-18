import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_update_consumer(self):
    consumer_ref = self._create_single_consumer()
    update_ref = {'consumer': {'description': uuid.uuid4().hex}}
    PROVIDERS.oauth_api.update_consumer(consumer_ref['id'], update_ref)
    self._assert_notify_sent(consumer_ref['id'], test_notifications.UPDATED_OPERATION, 'OS-OAUTH1:consumer')
    self._assert_last_audit(consumer_ref['id'], test_notifications.UPDATED_OPERATION, 'OS-OAUTH1:consumer', cadftaxonomy.SECURITY_ACCOUNT)