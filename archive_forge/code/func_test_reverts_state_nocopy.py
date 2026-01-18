import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_reverts_state_nocopy(self):
    wrapper = mock.MagicMock()
    task = import_flow._VerifyImageState(TASK_ID1, TASK_TYPE, wrapper, 'glance-direct')
    task.revert(mock.sentinel.result)
    action = wrapper.__enter__.return_value
    action.set_image_attribute.assert_called_once_with(status='queued')