from __future__ import (absolute_import, division, print_function)
def api_error(api, error):
    """format error message for api error, if error is present"""
    return 'calling: %s: got %s.' % (api, error) if error is not None else None