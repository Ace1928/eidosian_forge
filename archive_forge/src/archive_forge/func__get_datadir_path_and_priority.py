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
def _get_datadir_path_and_priority(self, datadir):
    """
        Gets directory paths and its priority from
        filesystem_store_datadirs option in glance-api.conf.

        :param datadir: is directory path with its priority.
        :returns: datadir_path as directory path
                 priority as priority associated with datadir_path
        :raises: BadStoreConfiguration exception if priority is invalid or
               empty directory path is specified.
        """
    priority = 0
    parts = [part.strip() for part in datadir.rsplit(':', 1)]
    datadir_path = parts[0]
    if len(parts) == 2 and parts[1]:
        try:
            priority = int(parts[1])
        except ValueError:
            msg = _('Invalid priority value %(priority)s in filesystem configuration') % {'priority': priority}
            LOG.exception(msg)
            raise exceptions.BadStoreConfiguration(store_name='filesystem', reason=msg)
    if not datadir_path:
        msg = _('Invalid directory specified in filesystem configuration')
        LOG.exception(msg)
        raise exceptions.BadStoreConfiguration(store_name='filesystem', reason=msg)
    return (datadir_path, priority)