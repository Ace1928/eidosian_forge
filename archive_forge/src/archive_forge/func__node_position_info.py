from __future__ import (absolute_import, division, print_function)
from yaml.constructor import SafeConstructor, ConstructorError
from yaml.nodes import MappingNode
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.parsing.yaml.objects import AnsibleMapping, AnsibleSequence, AnsibleUnicode, AnsibleVaultEncryptedUnicode
from ansible.parsing.vault import VaultLib
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var
def _node_position_info(self, node):
    column = node.start_mark.column + 1
    line = node.start_mark.line + 1
    datasource = self._ansible_file_name or node.start_mark.name
    return (datasource, line, column)