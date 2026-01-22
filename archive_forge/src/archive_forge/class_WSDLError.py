from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
class WSDLError(WSDLException):

    def __init__(self, origin, error_code, message):
        super(WSDLError, self).__init__('{0} ({1}): {2}'.format(origin, error_code, message))
        self.error_origin = origin
        self.error_code = error_code
        self.error_message = message