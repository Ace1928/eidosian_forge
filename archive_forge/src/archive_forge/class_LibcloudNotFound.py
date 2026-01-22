from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib  # noqa: F401, pylint: disable=unused-import
from ansible.module_utils.six.moves import configparser
from os.path import expanduser
from uuid import UUID
class LibcloudNotFound(Exception):
    """
    Exception raised when Apache libcloud cannot be found.
    """
    pass