from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def __append_whiteblank_per_line(self, blob, num_of_blank):
    ret = ' ' * num_of_blank
    ret += blob.replace('\n', '\n%s' % (' ' * num_of_blank))
    return ret