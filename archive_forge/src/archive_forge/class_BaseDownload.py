import abc
import glance_store as store_api
from glance_store import backend
from oslo_config import cfg
from oslo_log import log as logging
from taskflow import task
from glance.common import exception
from glance.i18n import _, _LE
class BaseDownload(task.Task, metaclass=abc.ABCMeta):
    default_provides = 'file_uri'

    def __init__(self, task_id, task_type, action_wrapper, stores, plugin_name):
        self.task_id = task_id
        self.task_type = task_type
        self.image_id = action_wrapper.image_id
        self.action_wrapper = action_wrapper
        self.stores = stores
        self._path = None
        self.plugin_name = plugin_name or 'Download'
        super(BaseDownload, self).__init__(name='%s-%s-%s' % (task_type, self.plugin_name, task_id))
        if CONF.enabled_backends:
            self.store = store_api.get_store_from_store_identifier('os_glance_staging_store')
        else:
            if CONF.node_staging_uri is None:
                msg = _('%(task_id)s of %(task_type)s not configured properly. Missing node_staging_uri: %(work_dir)s') % {'task_id': self.task_id, 'task_type': self.task_type, 'work_dir': CONF.node_staging_uri}
                raise exception.BadTaskConfiguration(msg)
            self.store = self._build_store()

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

    def revert(self, result, **kwargs):
        LOG.error(_LE('Task: %(task_id)s failed to import image %(image_id)s to the filesystem.'), {'task_id': self.task_id, 'image_id': self.image_id})
        with self.action_wrapper as action:
            action.set_image_attribute(status='queued')
            action.remove_importing_stores(self.stores)
            action.add_failed_stores(self.stores)
        if self._path is not None:
            LOG.debug('Deleting image %(image_id)s from staging area.', {'image_id': self.image_id})
            try:
                if CONF.enabled_backends:
                    store_api.delete(self._path, None)
                else:
                    store_api.delete_from_backend(self._path)
            except Exception:
                LOG.exception(_LE('Error reverting web/glance download task: %(task_id)s'), {'task_id': self.task_id})