from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def _handle_timeout(self, function, *args, **kwargs):
    retry = 0
    while True:
        try:
            return function(*args, **kwargs)
        except dns.exception.Timeout as exc:
            if retry >= self.timeout_retries:
                raise exc
            retry += 1