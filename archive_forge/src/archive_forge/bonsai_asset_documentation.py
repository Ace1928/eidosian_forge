from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import text_type
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
from ..module_utils import bonsai, errors

    Make sure that required values are not None and that if the value is
    present, it is of the correct type.
    