from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
def test_stacked_retries_dont_explode_retry_count(self):
    context = mock.Mock()
    context.session.is_active = False
    e = db_exc.DBConnectionError()
    mock.patch('time.sleep').start()
    with testtools.ExpectedException(db_exc.DBConnectionError):
        self._alt_context_function(context, db_api.MAX_RETRIES + 1, e)