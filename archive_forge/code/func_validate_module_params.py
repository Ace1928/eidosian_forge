from __future__ import absolute_import, division, print_function
from . import utils
def validate_module_params(params):
    if params['state'] == 'present':
        if not params['rules']:
            return 'state is present but all of the following are missing: rules'
    return None