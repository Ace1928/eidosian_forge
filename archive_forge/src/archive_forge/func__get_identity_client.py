import os
from heat.common.i18n import _
from heatclient import client as heat_client
from keystoneauth1.identity.generic import password
from keystoneauth1 import session
from keystoneclient.v3 import client as kc_v3
from novaclient import client as nova_client
from swiftclient import client as swift_client
def _get_identity_client(self):
    user_domain_id = self.conf.user_domain_id
    project_domain_id = self.conf.project_domain_id
    user_domain_name = self.conf.user_domain_name
    project_domain_name = self.conf.project_domain_name
    kwargs = {'username': self._username(), 'password': self._password(), 'project_name': self._project_name(), 'auth_url': self.conf.auth_url}
    if self.auth_version == '3':
        kwargs.update({'user_domain_id': user_domain_id, 'project_domain_id': project_domain_id, 'user_domain_name': user_domain_name, 'project_domain_name': project_domain_name})
    auth = password.Password(**kwargs)
    if self.insecure:
        verify_cert = False
    else:
        verify_cert = self.ca_file or True
    return KeystoneWrapperClient(auth, verify_cert)