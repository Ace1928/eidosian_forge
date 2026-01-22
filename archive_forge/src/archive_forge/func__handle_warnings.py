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
def _handle_warnings(self, res):
    """ display warnings, if enabled and any exist in the result """
    if C.ACTION_WARNINGS:
        if 'warnings' in res and res['warnings']:
            for warning in res['warnings']:
                self._display.warning(warning)
            del res['warnings']
        if 'deprecations' in res and res['deprecations']:
            for warning in res['deprecations']:
                self._display.deprecated(**warning)
            del res['deprecations']