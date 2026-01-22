from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
@property
def _dns_manager(self):
    if not self._dns_manager_attr:
        try:
            namespace = self._DNS_NAMESPACE % self._host
            self._dns_manager_attr = self._get_wmi_obj(namespace)
        except Exception:
            raise exceptions.DNSException(_('Namespace %(namespace)s not found. Make sure DNS Server feature is installed.') % {'namespace': namespace})
    return self._dns_manager_attr