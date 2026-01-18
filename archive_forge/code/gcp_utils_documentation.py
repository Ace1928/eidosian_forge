from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text

        This should be used for calling the GCP list APIs. It will return
        an array of items

        This takes a callback to a `return_if_object(module, response)`
        function that will decode the response + return a dictionary. Some
        modules handle the decode + error processing differently, so we should
        defer to the module to handle this.
        