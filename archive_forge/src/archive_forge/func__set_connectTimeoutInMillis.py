import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _set_connectTimeoutInMillis(self, config_store, value):
    self._update_provider(config_store, 'connect_timeout', value / 1000)