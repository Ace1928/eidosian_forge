from __future__ import (absolute_import, division, print_function)
import difflib
import json
import re
import sys
import textwrap
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import text_type
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.parsing.yaml.objects import AnsibleUnicode
from ansible.plugins import AnsiblePlugin
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import AnsibleUnsafeText, NativeJinjaUnsafeText, _is_unsafe
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
import yaml
def _pretty_represent_str(self, data):
    """Uses block style for multi-line strings"""
    if _is_unsafe(data):
        data = data._strip_unsafe()
    data = text_type(data)
    if _should_use_block(data):
        style = '|'
        if self._lossy:
            data = _munge_data_for_lossy_yaml(data)
    else:
        style = self.default_style
    node = yaml.representer.ScalarNode('tag:yaml.org,2002:str', data, style=style)
    if self.alias_key is not None:
        self.represented_objects[self.alias_key] = node
    return node