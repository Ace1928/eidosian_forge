import logging
import os
import time
from . import docker_base as base
def _create_config_os_ken(self):
    c = base.CmdBuffer()
    c << '[DEFAULT]'
    c << 'verbose=True'
    c << 'log_file=/etc/os_ken/manager.log'
    with open(self.OSKEN_CONF, 'w') as f:
        LOG.info("[%s's new config]" % self.name)
        LOG.info(str(c))
        f.writelines(str(c))