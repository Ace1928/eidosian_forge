import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
class _VerifyStaging(task.Task):
    default_provides = 'file_path'

    def __init__(self, task_id, task_type, task_repo, uri):
        self.task_id = task_id
        self.task_type = task_type
        self.task_repo = task_repo
        self.uri = uri
        super(_VerifyStaging, self).__init__(name='%s-ConfigureStaging-%s' % (task_type, task_id))
        try:
            uri.index('file:///', 0)
        except ValueError:
            msg = _("%(task_id)s of %(task_type)s not configured properly. Value of node_staging_uri must be  in format 'file://<absolute-path>'") % {'task_id': self.task_id, 'task_type': self.task_type}
            raise exception.BadTaskConfiguration(msg)
        if not CONF.enabled_backends:
            self._build_store()

    def _build_store(self):
        conf = cfg.ConfigOpts()
        try:
            backend.register_opts(conf)
        except cfg.DuplicateOptError:
            pass
        conf.set_override('filesystem_store_datadir', CONF.node_staging_uri[7:], group='glance_store')
        store = backend._load_store(conf, 'file')
        try:
            store.configure()
        except AttributeError:
            msg = _('%(task_id)s of %(task_type)s not configured properly. Could not load the filesystem store') % {'task_id': self.task_id, 'task_type': self.task_type}
            raise exception.BadTaskConfiguration(msg)

    def execute(self):
        """Test the backend store and return the 'file_path'"""
        return self.uri