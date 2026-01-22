from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class ReplicationSessionFailoverHandler:

    def handle(self, session_object, session_params, replication_session_obj):
        if replication_session_obj and session_params['state'] == 'present' and (session_params['failover_with_sync'] is not None):
            session_object.result['changed'] = session_object.failover(replication_session_obj, session_params['failover_with_sync'], session_params['force']) or False
            if session_object.result['changed']:
                replication_session_obj = session_object.get_replication_session(session_params['session_id'], session_params['session_name'])
        ReplicationSessionFailbackHandler().handle(session_object, session_params, replication_session_obj)