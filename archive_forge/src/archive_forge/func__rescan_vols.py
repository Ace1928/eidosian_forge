import json
import os
import urllib
from oslo_log import log as logging
import requests
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.privileged import scaleio as priv_scaleio
from os_brick import utils
def _rescan_vols(self):
    LOG.info('ScaleIO rescan volumes')
    try:
        priv_scaleio.rescan_vols(self.RESCAN_VOLS_OP_CODE)
    except (IOError, OSError) as e:
        msg = _('Error querying volumes: %s') % e
        LOG.error(msg)
        raise exception.BrickException(message=msg)