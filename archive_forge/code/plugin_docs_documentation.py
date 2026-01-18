from __future__ import (absolute_import, division, print_function)
import ast
import tokenize
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display

    Quickly find short_description using string methods instead of node parsing.
    This does not return a full set of documentation strings and is intended for
    operations like ansible-doc -l.
    