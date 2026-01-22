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
@db_api.retry_if_session_inactive('alt_context')
def _alt_context_function(self, alt_context, *args, **kwargs):
    return self._decorated_function(*args, **kwargs)