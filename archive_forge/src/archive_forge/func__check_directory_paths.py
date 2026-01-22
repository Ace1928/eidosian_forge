import errno
import logging
import os
import stat
import urllib
import jsonschema
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LW
import glance_store.location
from the filesystem store. The users running the services that are
def _check_directory_paths(self, datadir_path, directory_paths, priority_paths):
    """
        Checks if directory_path is already present in directory_paths.

        :datadir_path is directory path.
        :datadir_paths is set of all directory paths.
        :raises: BadStoreConfiguration exception if same directory path is
               already present in directory_paths.
        """
    if datadir_path in directory_paths:
        msg = _('Directory %(datadir_path)s specified multiple times in filesystem_store_datadirs option of filesystem configuration') % {'datadir_path': datadir_path}
        if datadir_path not in priority_paths:
            LOG.exception(msg)
            raise exceptions.BadStoreConfiguration(store_name='filesystem', reason=msg)
        LOG.warning(msg)