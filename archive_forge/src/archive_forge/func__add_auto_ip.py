import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _add_auto_ip(self, server, wait=False, timeout=60, reuse=True):
    skip_attach = False
    created = False
    if reuse:
        f_ip = self.available_floating_ip(server=server)
    else:
        start_time = time.time()
        f_ip = self.create_floating_ip(server=server, wait=wait, timeout=timeout)
        timeout = timeout - (time.time() - start_time)
        if server:
            skip_attach = True
        created = True
    try:
        return self._attach_ip_to_server(server=server, floating_ip=f_ip, wait=wait, timeout=timeout, skip_attach=skip_attach)
    except exceptions.ResourceTimeout:
        if self._use_neutron_floating() and created:
            self.log.error('Timeout waiting for floating IP to become active. Floating IP %(ip)s:%(id)s was created for server %(server)s but is being deleted due to activation failure.', {'ip': f_ip['floating_ip_address'], 'id': f_ip['id'], 'server': server['id']})
            try:
                self.delete_floating_ip(f_ip['id'])
            except Exception as e:
                self.log.error('FIP LEAK: Attempted to delete floating ip %(fip)s but received %(exc)s exception: %(err)s', {'fip': f_ip['id'], 'exc': e.__class__, 'err': str(e)})
                raise e
        raise