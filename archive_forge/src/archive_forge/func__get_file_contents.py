from __future__ import (absolute_import, division, print_function)
import copy
import os
import os.path
import re
import tempfile
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleFileNotFound, AnsibleParserError
from ansible.module_utils.basic import is_executable
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.quoting import unquote
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.vault import VaultLib, b_HEADER, is_encrypted, is_encrypted_file, parse_vaulttext_envelope, PromptVaultSecret
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
def _get_file_contents(self, file_name: str) -> tuple[bytes, bool]:
    """
        Reads the file contents from the given file name

        If the contents are vault-encrypted, it will decrypt them and return
        the decrypted data

        :arg file_name: The name of the file to read.  If this is a relative
            path, it will be expanded relative to the basedir
        :raises AnsibleFileNotFound: if the file_name does not refer to a file
        :raises AnsibleParserError: if we were unable to read the file
        :return: Returns a byte string of the file contents
        """
    if not file_name or not isinstance(file_name, (binary_type, text_type)):
        raise AnsibleParserError("Invalid filename: '%s'" % to_native(file_name))
    b_file_name = to_bytes(self.path_dwim(file_name))
    if not self.path_exists(b_file_name):
        raise AnsibleFileNotFound('Unable to retrieve file contents', file_name=file_name)
    try:
        with open(b_file_name, 'rb') as f:
            data = f.read()
            return self._decrypt_if_vault_data(data, b_file_name)
    except (IOError, OSError) as e:
        raise AnsibleParserError("an error occurred while trying to read the file '%s': %s" % (file_name, to_native(e)), orig_exc=e)