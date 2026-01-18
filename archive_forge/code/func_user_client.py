import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
@property
def user_client(self):
    if not hasattr(self, '_user_client'):
        self._user_client = self.get_user_client()
    return self._user_client