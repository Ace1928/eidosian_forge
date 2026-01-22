from __future__ import absolute_import, division, print_function
import re
import shlex
import time
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import string_types, text_type
from ansible.module_utils.six.moves import zip
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
class AddConditionError(Exception):

    def __init__(self, msg, condition):
        super(AddConditionError, self).__init__(msg)
        self.condition = condition