from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
import ssl
import traceback
def wrap_context(*args, **kwargs):
    kwargs.pop('server_hostname', None)
    return ctx.wrap_socket(*args, server_hostname=host, **kwargs)