import os
from heat.common.i18n import _
from heatclient import client as heat_client
from keystoneauth1.identity.generic import password
from keystoneauth1 import session
from keystoneclient.v3 import client as kc_v3
from novaclient import client as nova_client
from swiftclient import client as swift_client
def _get_compute_client(self):
    return nova_client.Client(self.NOVA_API_VERSION, session=self.identity_client.session, service_type='compute', endpoint_type='publicURL', region_name=self.conf.region, os_cache=False, http_log_debug=True)