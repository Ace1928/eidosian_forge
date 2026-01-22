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
def _serialize_diff(self, diff):
    try:
        result_format = self.get_option('result_format')
    except KeyError:
        result_format = 'json'
    try:
        pretty_results = self.get_option('pretty_results')
    except KeyError:
        pretty_results = None
    if result_format == 'json':
        return json.dumps(diff, sort_keys=True, indent=4, separators=(u',', u': ')) + u'\n'
    elif result_format == 'yaml':
        lossy = pretty_results in (None, True)
        return '%s\n' % textwrap.indent(yaml.dump(diff, allow_unicode=True, Dumper=_AnsibleCallbackDumper(lossy=lossy), default_flow_style=False, indent=4), '    ')