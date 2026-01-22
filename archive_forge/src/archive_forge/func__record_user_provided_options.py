import copy
from botocore.compat import OrderedDict
from botocore.endpoint import DEFAULT_TIMEOUT, MAX_POOL_CONNECTIONS
from botocore.exceptions import (
def _record_user_provided_options(self, args, kwargs):
    option_order = list(self.OPTION_DEFAULTS)
    user_provided_options = {}
    for key, value in kwargs.items():
        if key in self.OPTION_DEFAULTS:
            user_provided_options[key] = value
        else:
            raise TypeError(f"Got unexpected keyword argument '{key}'")
    if len(args) > len(option_order):
        raise TypeError(f'Takes at most {len(option_order)} arguments ({len(args)} given)')
    for i, arg in enumerate(args):
        if option_order[i] in user_provided_options:
            raise TypeError(f"Got multiple values for keyword argument '{option_order[i]}'")
        user_provided_options[option_order[i]] = arg
    return user_provided_options