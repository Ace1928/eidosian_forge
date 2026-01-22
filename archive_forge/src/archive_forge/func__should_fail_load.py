import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def _should_fail_load(self, config_loader, reason):
    try:
        config_loader.load_and_set()
        self.fail('Should fail because %s' % reason)
    except ConfigException:
        pass