import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_update_port_on_error(self):
    core_plugin = mock.Mock()
    with mock.patch.object(excutils, 'save_and_reraise_exception'):
        with mock.patch.object(utils, 'LOG'):
            with utils.update_port_on_error(core_plugin, 'ctx', '1', '2'):
                raise Exception()
    core_plugin.update_port.assert_called_once_with('ctx', '1', {'port': '2'})