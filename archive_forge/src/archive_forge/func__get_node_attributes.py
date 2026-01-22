from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
@staticmethod
def _get_node_attributes(node):
    attr = {}
    if 'attributes' in node.obj_dict:
        attr.update(node.obj_dict['attributes'])
    pos = node.get_pos()
    if pos is not None:
        pos = remove_quotes(pos)
        xx, yy = pos.split(',')
        attr['pos'] = (float(xx), float(yy))
    return attr