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
def test_wrapper_failure(self):
    mock_repo = mock.MagicMock()
    mock_repo.get.return_value.extra_properties = {'os_glance_import_task': TASK_ID1}
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)

    class SpecificError(Exception):
        pass
    try:
        with wrapper:
            raise SpecificError('some failure')
    except SpecificError:
        pass
    mock_repo.get.assert_called_once_with(IMAGE_ID1)
    mock_repo.save.assert_not_called()