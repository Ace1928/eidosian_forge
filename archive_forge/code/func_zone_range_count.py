from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def zone_range_count(client, tag):
    """
    Returns the count of records that exists for the given tag in config.tags
    """
    return client['config'].tags.count_documents({'tag': tag})