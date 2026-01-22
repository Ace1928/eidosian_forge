from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import sys
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import VaultEditor, VaultLib, match_encrypt_secret
from ansible.utils.display import Display
def _format_output_vault_strings(self, b_plaintext_list, vault_id=None):
    show_delimiter = False
    if len(b_plaintext_list) > 1:
        show_delimiter = True
    output = []
    for index, b_plaintext_info in enumerate(b_plaintext_list):
        b_plaintext, src, name = b_plaintext_info
        b_ciphertext = self.editor.encrypt_bytes(b_plaintext, self.encrypt_secret, vault_id=vault_id)
        yaml_text = self.format_ciphertext_yaml(b_ciphertext, name=name)
        err_msg = None
        if show_delimiter:
            human_index = index + 1
            if name:
                err_msg = '# The encrypted version of variable ("%s", the string #%d from %s).\n' % (name, human_index, src)
            else:
                err_msg = '# The encrypted version of the string #%d from %s.)\n' % (human_index, src)
        output.append({'out': yaml_text, 'err': err_msg})
    return output