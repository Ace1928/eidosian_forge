import pkgutil
import sys
from oslo_concurrency import lockutils
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import driver
from stevedore import enabled
from neutron_lib._i18n import _
def _add_extension(self, ext):
    if ext.name in self._extensions:
        msg = _("Plugin '%(p)s' already in namespace: %(ns)s") % {'p': ext.name, 'ns': self.namespace}
        raise KeyError(msg)
    LOG.debug("Loaded plugin '%s' from namespace: %s", ext.name, self.namespace)
    self._extensions[ext.name] = ext