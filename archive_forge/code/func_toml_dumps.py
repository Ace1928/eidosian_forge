from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections.abc import MutableMapping, MutableSequence
from functools import partial
from ansible.errors import AnsibleFileNotFound, AnsibleParserError, AnsibleRuntimeError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import string_types, text_type
from ansible.parsing.yaml.objects import AnsibleSequence, AnsibleUnicode
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import AnsibleUnsafeBytes, AnsibleUnsafeText
def toml_dumps(data):
    if HAS_TOML:
        return toml.dumps(convert_yaml_objects_to_native(data))
    elif HAS_TOMLIW:
        return tomli_w.dumps(convert_yaml_objects_to_native(data))
    raise AnsibleRuntimeError('The python "toml" or "tomli-w" library is required when using the TOML output format')