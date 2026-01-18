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
@mock.patch.object(import_flow, 'LOG')
def test_remove_location_for_store_pop_failures(self, mock_log):

    class TestList(list):

        def pop(self):
            pass
    self.image.locations = TestList([{'metadata': {'store': 'foo'}}])
    with mock.patch.object(self.image.locations, 'pop', new_callable=mock.PropertyMock) as mock_pop:
        mock_pop.side_effect = store_exceptions.NotFound(image='image')
        self.actions.remove_location_for_store('foo')
        mock_log.warning.assert_called_once_with(_('Error deleting from store foo when reverting.'))
        mock_log.warning.reset_mock()
        mock_pop.side_effect = store_exceptions.Forbidden()
        self.actions.remove_location_for_store('foo')
        mock_log.warning.assert_called_once_with(_('Error deleting from store foo when reverting.'))
        mock_log.warning.reset_mock()
        mock_pop.side_effect = Exception
        self.actions.remove_location_for_store('foo')
        mock_log.warning.assert_called_once_with(_('Unexpected exception when deleting from store foo.'))
        mock_log.warning.reset_mock()