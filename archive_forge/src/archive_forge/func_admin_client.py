import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
@property
def admin_client(self):
    if not hasattr(self, '_admin_client'):
        self._admin_client = self.get_admin_client()
    return self._admin_client