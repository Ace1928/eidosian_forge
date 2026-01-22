import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def _build_mock_barbican(self):
    self.mock_barbican = mock.MagicMock(name='mock_barbican')
    self.get = self.mock_barbican.secrets.get
    self.delete = self.mock_barbican.secrets.delete
    self.store = self.mock_barbican.secrets.store
    self.create = self.mock_barbican.secrets.create
    self.list = self.mock_barbican.secrets.list
    self.add_consumer = self.mock_barbican.secrets.add_consumer
    self.remove_consumer = self.mock_barbican.secrets.remove_consumer
    self.list_versions = self.mock_barbican.versions.list_versions
    self.key_mgr._barbican_client = self.mock_barbican
    self.key_mgr._current_context = self.ctxt
    self.key_mgr._version_client = self.mock_barbican