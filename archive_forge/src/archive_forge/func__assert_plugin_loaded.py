import pkgutil
import sys
from oslo_concurrency import lockutils
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import driver
from stevedore import enabled
from neutron_lib._i18n import _
def _assert_plugin_loaded(self, plugin_name):
    if plugin_name not in self._extensions:
        msg = _("Plugin '%(p)s' not in namespace: %(ns)s") % {'p': plugin_name, 'ns': self.namespace}
        raise KeyError(msg)