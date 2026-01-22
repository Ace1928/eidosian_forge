from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
def _decode_wsdl_array(result, node, root_ns, ids):
    for item in node:
        if item.tag != 'item':
            raise WSDLCodingException('Invalid child tag "{0}" in map!'.format(item.tag))
        result.append(decode_wsdl(item, root_ns, ids))