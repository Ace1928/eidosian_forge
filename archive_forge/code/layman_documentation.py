from __future__ import absolute_import, division, print_function
import shutil
import traceback
from os import path
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
Synchronize all of the installed overlays.

    :raises ModuleError
    