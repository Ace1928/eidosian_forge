from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
def _create_envelope(self, tag, **kwarg):
    return self._create(tag, self._main_ns, **kwarg)