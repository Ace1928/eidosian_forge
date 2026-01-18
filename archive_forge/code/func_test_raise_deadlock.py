import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_raise_deadlock(self):

    class TestException(Exception):
        pass
    self.attempts = 3

    def _mock_get_session():

        def _raise_exceptions():
            self.attempts -= 1
            if self.attempts <= 0:
                raise TestException('Exit')
            raise db_exc.DBDeadlock('Fake Exception')
        return _raise_exceptions
    with mock.patch.object(api, 'get_session') as sess:
        sess.side_effect = _mock_get_session()
        try:
            api.image_update(None, 'fake-id', {})
        except TestException:
            self.assertEqual(3, sess.call_count)
    self.attempts = 3
    with mock.patch.object(api, 'get_session') as sess:
        sess.side_effect = _mock_get_session()
        try:
            api.image_destroy(None, 'fake-id')
        except TestException:
            self.assertEqual(3, sess.call_count)