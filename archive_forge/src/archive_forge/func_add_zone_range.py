from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def add_zone_range(client, namespace, min, max, tag):
    """
    Adds a zone range
    @client - MongoDB connection
    @namespace - In the form database.collection
    @min - The min range value
    @max - The max range value
    @tag - The tag or Zone name
    """
    cmd_doc = OrderedDict([('updateZoneKeyRange', namespace), ('min', min), ('max', max), ('zone', tag)])
    client['admin'].command(cmd_doc)