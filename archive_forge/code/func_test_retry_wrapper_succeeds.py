from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_retry_wrapper_succeeds(self):

    @api.wrap_db_retry(max_retries=10)
    def some_method():
        pass
    some_method()