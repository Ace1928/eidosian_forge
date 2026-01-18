from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import open_url
from ansible.plugins.callback import CallbackBase

        Display info about playbook statistics
        