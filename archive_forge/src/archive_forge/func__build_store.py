import abc
import glance_store as store_api
from glance_store import backend
from oslo_config import cfg
from oslo_log import log as logging
from taskflow import task
from glance.common import exception
from glance.i18n import _, _LE
def _build_store(self):
    conf = cfg.ConfigOpts()
    try:
        backend.register_opts(conf)
    except cfg.DuplicateOptError:
        pass
    conf.set_override('filesystem_store_datadir', CONF.node_staging_uri[7:], group='glance_store')
    store = store_api.backend._load_store(conf, 'file')
    if store is None:
        msg = _('%(task_id)s of %(task_type)s not configured properly. Could not load the filesystem store') % {'task_id': self.task_id, 'task_type': self.task_type}
        raise exception.BadTaskConfiguration(msg)
    store.configure()
    return store