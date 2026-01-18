from __future__ import (absolute_import, division, print_function)
import re
import time
import json
import os
from itertools import chain
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible.plugins.cliconf import CliconfBase, enable_mode
from ansible.module_utils.common._collections_compat import Mapping

        Edit banner on remote device
        :param banners: Banners to be loaded in json format
        :param multiline_delimiter: Line delimiter for banner
        :param commit: Boolean value that indicates if the device candidate
               configuration should be  pushed in the running configuration or discarded.
        :param diff: Boolean flag to indicate if configuration that is applied on remote host should
                     generated and returned in response or not
        :return: Returns response of executing the configuration command received
             from remote host
        