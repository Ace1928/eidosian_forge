import os
from heat.common.i18n import _
from heatclient import client as heat_client
from keystoneauth1.identity.generic import password
from keystoneauth1 import session
from keystoneclient.v3 import client as kc_v3
from novaclient import client as nova_client
from swiftclient import client as swift_client
class KeystoneWrapperClient(object):
    """Wrapper object for keystone client

    This wraps keystone client, so we can encpasulate certain
    added properties like auth_token, project_id etc.
    """

    def __init__(self, auth_plugin, verify=True):
        self.auth_plugin = auth_plugin
        self.session = session.Session(auth=auth_plugin, verify=verify)

    @property
    def auth_token(self):
        return self.auth_plugin.get_token(self.session)

    @property
    def auth_ref(self):
        return self.auth_plugin.get_access(self.session)

    @property
    def project_id(self):
        return self.auth_plugin.get_project_id(self.session)

    def get_endpoint_url(self, service_type, region=None):
        kwargs = {'service_type': service_type, 'region_name': region}
        return self.auth_ref.service_catalog.url_for(**kwargs)