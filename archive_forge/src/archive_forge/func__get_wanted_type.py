from __future__ import absolute_import, division, print_function
import abc
import copy
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleFallbackNotFound, SEQUENCETYPE, remove_values
from ansible.module_utils.common._collections_compat import (
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.validation import (
from ansible.module_utils.common.text.formatters import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.action import ActionBase
def _get_wanted_type(self, wanted, k):
    if not callable(wanted):
        if wanted is None:
            wanted = 'str'
        try:
            type_checker = self._CHECK_ARGUMENT_TYPES_DISPATCHER[wanted]
        except KeyError:
            self.fail_json(msg='implementation error: unknown type %s requested for %s' % (wanted, k))
    else:
        type_checker = wanted
        wanted = getattr(wanted, '__name__', to_native(type(wanted)))
    return (type_checker, wanted)