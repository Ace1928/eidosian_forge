from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_bytes
class TagClass:
    universal = 0
    application = 1
    context_specific = 2
    private = 3