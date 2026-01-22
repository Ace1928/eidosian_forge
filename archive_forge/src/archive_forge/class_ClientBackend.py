import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import enabled
from heat.common import exception
from heat.common.i18n import _
from heat.common import pluginutils
class ClientBackend(object):
    """Class for delaying choosing the backend client module.

    Delay choosing the backend client module until the client's class needs
    to be initialized.
    """

    def __new__(cls, context):
        if cfg.CONF.cloud_backend == _default_backend:
            return OpenStackClients(context)
        else:
            try:
                return importutils.import_object(cfg.CONF.cloud_backend, context)
            except (ImportError, RuntimeError, cfg.NoSuchOptError) as err:
                msg = _('Invalid cloud_backend setting in heat.conf detected - %s') % str(err)
                LOG.error(msg)
                raise exception.Invalid(reason=msg)