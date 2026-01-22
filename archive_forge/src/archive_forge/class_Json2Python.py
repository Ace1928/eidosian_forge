from __future__ import (absolute_import, division, print_function)
import ast
from itertools import islice, chain
from types import GeneratorType
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils.native_jinja import NativeJinjaText
class Json2Python(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id not in _JSON_MAP:
            return node
        return ast.Constant(value=_JSON_MAP[node.id])