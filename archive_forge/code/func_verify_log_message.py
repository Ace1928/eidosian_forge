import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def verify_log_message(self, data):
    """Verify log message.

        Tests that use this are a little brittle because adding more
        logging can break them.

        TODO(dstanek): remove the need for this in a future refactoring

        """
    log_fn = self.mock_log.debug
    self.assertEqual(len(data), log_fn.call_count)
    for datum in data:
        log_fn.assert_any_call(mock.ANY, datum)