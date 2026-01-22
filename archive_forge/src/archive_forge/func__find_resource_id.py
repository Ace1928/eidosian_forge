from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
@os_client.MEMOIZE_FINDER
def _find_resource_id(self, tenant_id, resource, name_or_id, cmd_resource):
    return neutronV20.find_resourceid_by_name_or_id(self.client(), resource, name_or_id, cmd_resource=cmd_resource)