from __future__ import (absolute_import, division, print_function)
def unexpected_response_error(api, response, query=None):
    """format error message for reponse not matching expectations"""
    msg = 'calling: %s: unexpected response %s.' % (api, repr(response))
    if query:
        msg += ' for query: %s' % repr(query)
    return (response, msg)