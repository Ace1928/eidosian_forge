from __future__ import (absolute_import, division, print_function)
import base64
import os
import json
from stat import S_IRUSR, S_IWUSR
from ansible import constants as C
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
class NoTokenSentinel(object):
    """ Represents an ansible.cfg server with not token defined (will ignore cmdline and GALAXY_TOKEN_PATH. """

    def __new__(cls, *args, **kwargs):
        return cls