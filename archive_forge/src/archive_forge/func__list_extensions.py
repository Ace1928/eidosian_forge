from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
@os_client.MEMOIZE_EXTENSIONS
def _list_extensions(self):
    extensions = self.client().list_extensions().get('extensions')
    return set((extension.get('alias') for extension in extensions))