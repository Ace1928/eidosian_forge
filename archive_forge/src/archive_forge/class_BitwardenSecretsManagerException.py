from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.errors import AnsibleLookupError
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.ajson import AnsibleJSONDecoder
from ansible.plugins.lookup import LookupBase
class BitwardenSecretsManagerException(AnsibleLookupError):
    pass