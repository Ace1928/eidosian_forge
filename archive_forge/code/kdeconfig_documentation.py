from __future__ import (absolute_import, division, print_function)
import os
import shutil
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
Invoke kwriteconfig with arguments