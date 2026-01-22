from __future__ import (absolute_import, division, print_function)
import ast
from itertools import islice, chain
from types import GeneratorType
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils.native_jinja import NativeJinjaText
def ansible_concat(nodes):
    """Return a string of concatenated compiled nodes. Throw an undefined error
    if any of the nodes is undefined. Other than that it is equivalent to
    Jinja2's default concat function.

    Used in Templar.template() when jinja2_native=False and convert_data=False.
    """
    return ''.join([to_text(v) for v in nodes])