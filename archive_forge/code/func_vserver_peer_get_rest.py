from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_get_rest(self, target):
    """
        Get current vserver peer info
        :return: Dictionary of current vserver peer details if query successful, else return None
        """
    api = 'svm/peers'
    vserver_info = {}
    vserver, remote_vserver = self.get_local_and_peer_vserver(target)
    restapi = self.rest_api if target == 'source' else self.dst_rest_api
    options = {'svm.name': vserver, 'peer.svm.name': remote_vserver, 'fields': 'name,svm.name,peer.svm.name,state,uuid'}
    if target == 'peer' and self.peer_relation_uuid is not None:
        options['uuid'] = self.peer_relation_uuid
    record, error = rest_generic.get_one_record(restapi, api, options)
    if error:
        self.module.fail_json(msg='Error fetching vserver peer %s: %s' % (self.parameters['vserver'], error))
    if record is not None:
        vserver_info['vserver'] = self.na_helper.safe_get(record, ['svm', 'name'])
        vserver_info['peer_vserver'] = self.na_helper.safe_get(record, ['peer', 'svm', 'name'])
        vserver_info['peer_state'] = record.get('state')
        vserver_info['local_peer_vserver_uuid'] = record.get('uuid')
        vserver_info['local_peer_vserver'] = record['name']
        return vserver_info
    return None