from __future__ import (absolute_import, division, print_function)
import traceback
import json
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.plugins.inventory import (
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _set_credentials(self):
    """
            :param config_data: contents of the inventory config file
        """
    self.client_id = self.get_option('client_id')
    self.client_secret = self.get_option('client_secret')