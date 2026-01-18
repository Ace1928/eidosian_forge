from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import utils
def uptime(self, hypervisor):
    """
        Get the uptime for a specific hypervisor.

        :param hypervisor: Either a Hypervisor object or an ID. Starting with
            microversion 2.53 the ID must be a UUID value.
        """
    if self.api_version < api_versions.APIVersion('2.88'):
        return self._get('/os-hypervisors/%s/uptime' % base.getid(hypervisor), 'hypervisor')
    resp, body = self.api.client.get('/os-hypervisors/%s' % base.getid(hypervisor))
    content = {k: v for k, v in body['hypervisor'].items() if k in ('id', 'hypervisor_hostname', 'state', 'status', 'uptime')}
    return self.resource_class(self, content, loaded=True, resp=resp)