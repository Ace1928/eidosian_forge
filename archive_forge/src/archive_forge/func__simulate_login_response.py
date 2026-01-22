from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
def _simulate_login_response(self, token, lookup_response=None):
    """returns a similar structure to a login method's return, optionally incorporating a lookup-self response"""
    response = {'auth': {'client_token': token}}
    if lookup_response is None:
        return response
    response.update(lookup_response, auth=response['auth'])
    response['auth'].update(lookup_response['data'])
    metadata = response['auth'].pop('meta', None)
    if metadata:
        response['auth']['metadata'] = metadata
    return response