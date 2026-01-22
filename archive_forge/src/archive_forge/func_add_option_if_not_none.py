from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def add_option_if_not_none(param_name, module, connection_params):
    """
    @param_name - The parameter name to check
    @module - The ansible module object
    @connection_params - Dict containing the connection params
    """
    if module.params[param_name] is not None:
        connection_params[param_name] = module.params[param_name]
    return connection_params