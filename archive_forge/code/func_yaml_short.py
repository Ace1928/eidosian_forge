from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import yaml
import shlex
import subprocess
from collections.abc import Mapping
from ansible import context
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.config.manager import ConfigManager, Setting
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.six import string_types
from ansible.parsing.quoting import is_quoted
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def yaml_short(data):
    return yaml_dump(data, default_flow_style=True, default_style="''")