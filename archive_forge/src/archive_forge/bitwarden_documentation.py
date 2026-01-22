from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.ajson import AnsibleJSONDecoder
from ansible.plugins.lookup import LookupBase
Return a list of the specified field for records whose search_field match search_value
        and filtered by collection if collection has been provided.

        If field is None, return the whole record for each match.
        