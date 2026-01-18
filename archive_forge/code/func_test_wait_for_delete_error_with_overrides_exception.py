import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
@mock.patch.object(time, 'sleep')
def test_wait_for_delete_error_with_overrides_exception(self, mock_sleep):
    mock_get = mock.Mock(side_effect=Exception)
    manager = mock.MagicMock(get=mock_get)
    res_id = str(uuid.uuid4())
    self.assertTrue(utils.wait_for_delete(manager, res_id, exception_name=['Exception']))
    mock_sleep.assert_not_called()