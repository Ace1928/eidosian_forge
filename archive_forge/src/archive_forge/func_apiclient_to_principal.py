from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def apiclient_to_principal(fusion, api_client_key):
    """Given an API client issuer ID, such as "pure1:apikey:123xXxyYyzYzASDF",
    return the associated principal
    """
    id_api_instance = purefusion.IdentityManagerApi(fusion)
    api_clients = id_api_instance.list_users(name=api_client_key)
    if len(api_clients) > 0:
        return api_clients[0].id
    return None