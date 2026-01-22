from __future__ import (absolute_import, division, print_function)
import ast
from itertools import islice, chain
from types import GeneratorType
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils.native_jinja import NativeJinjaText
def ansible_native_concat(nodes):
    """Return a native Python type from the list of compiled nodes. If the
    result is a single node, its value is returned. Otherwise, the nodes are
    concatenated as strings. If the result can be parsed with
    :func:`ast.literal_eval`, the parsed value is returned. Otherwise, the
    string is returned.

    https://github.com/pallets/jinja/blob/master/src/jinja2/nativetypes.py
    """
    head = list(islice(nodes, 2))
    if not head:
        return None
    if len(head) == 1:
        out = head[0]
        if isinstance(out, AnsibleVaultEncryptedUnicode):
            return out.data
        if isinstance(out, NativeJinjaText):
            return out
        if not isinstance(out, string_types):
            return out
    else:
        if isinstance(nodes, GeneratorType):
            nodes = chain(head, nodes)
        out = ''.join([to_text(v) for v in nodes])
    try:
        evaled = ast.literal_eval(ast.parse(out, mode='eval'))
    except (TypeError, ValueError, SyntaxError, MemoryError):
        return out
    if isinstance(evaled, string_types):
        quote = out[0]
        return f'{quote}{evaled}{quote}'
    return evaled