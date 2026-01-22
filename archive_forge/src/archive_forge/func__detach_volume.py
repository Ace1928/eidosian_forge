import os
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick import utils
def _detach_volume(self, volume_name):
    return self._cli_cmd('detach', volume_name)