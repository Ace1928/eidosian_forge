from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def _get_user_provided_value(module, attribute_name):
    """
    Returns the user provided value for "attribute_name". We consider aliases in the module.
    """
    user_provided_value = module.params.get(attribute_name, None)
    if user_provided_value is None:
        option_alias_for_attribute = module.aliases.get(attribute_name, None)
        if option_alias_for_attribute is not None:
            user_provided_value = module.params.get(option_alias_for_attribute, None)
    return user_provided_value