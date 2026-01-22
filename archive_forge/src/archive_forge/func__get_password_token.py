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
@staticmethod
def _get_password_token(connection_properties):
    if 'serverPassword' in connection_properties:
        return (connection_properties['serverPassword'], connection_properties['serverToken'])
    LOG.info('Get ScaleIO connector password from configuration file')
    try:
        password = priv_scaleio.get_connector_password(CONNECTOR_CONF_PATH, connection_properties['config_group'], connection_properties.get('failed_over', False))
        return (password, None)
    except Exception as e:
        msg = _('Error getting ScaleIO connector password from configuration file: %s') % e
        LOG.error(msg)
        raise exception.BrickException(message=msg)