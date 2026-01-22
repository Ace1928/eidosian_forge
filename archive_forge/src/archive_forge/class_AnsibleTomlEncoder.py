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
class AnsibleTomlEncoder(toml.TomlEncoder):

    def __init__(self, *args, **kwargs):
        super(AnsibleTomlEncoder, self).__init__(*args, **kwargs)
        self.dump_funcs.update({AnsibleSequence: self.dump_funcs.get(list), AnsibleUnicode: self.dump_funcs.get(str), AnsibleUnsafeBytes: self.dump_funcs.get(str), AnsibleUnsafeText: self.dump_funcs.get(str)})