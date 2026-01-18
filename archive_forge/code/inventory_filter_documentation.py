from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types

    Determine whether a host should be accepted (``True``) or not (``False``).
    