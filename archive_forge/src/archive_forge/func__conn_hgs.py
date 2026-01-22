import re
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import hostutils
from oslo_log import log as logging
@property
def _conn_hgs(self):
    if not self._conn_hgs_attr:
        try:
            namespace = self._HGS_NAMESPACE % self._host
            self._conn_hgs_attr = self._get_wmi_conn(namespace)
        except Exception:
            raise exceptions.OSWinException(_('Namespace %(namespace)s is not supported on this Windows version.') % {'namespace': namespace})
    return self._conn_hgs_attr