import os
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick import utils
def _get_volume_path(self, connection_properties):
    out = self._query_attached_volume(connection_properties['volume_id'])
    if not out or int(out['ret_code']) != 0:
        msg = _("Couldn't find attached volume.")
        LOG.error(msg)
        raise exception.BrickException(message=msg)
    return out['dev_addr']