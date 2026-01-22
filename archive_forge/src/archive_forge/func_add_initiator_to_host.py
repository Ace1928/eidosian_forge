from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def add_initiator_to_host(self, host_details, initiators):
    """ Add initiator to host """
    try:
        existing_initiators = self.get_host_initiators_list(host_details)
        ' if current and exisitng initiators are same'
        if initiators and set(initiators).issubset(set(existing_initiators)):
            LOG.info('Initiators are already present in host: %s', host_details.name)
            return (False, host_details)
        ' get the list of non-mapped initiators out of the\n                given initiators'
        host_id = host_details.id
        unmapped_initiators = self.get_list_unmapped_initiators(initiators, host_id)
        ' if any of the Initiators is invalid or already mapped '
        if unmapped_initiators is None or len(unmapped_initiators) < len(initiators):
            error_message = 'Provide valid initiators.'
            LOG.error(error_message)
            self.module.fail_json(msg=error_message)
        LOG.info('Adding initiators to host %s', host_details.name)
        for id in unmapped_initiators:
            host_details.add_initiator(uid=id)
            updated_host = self.unity.get_host(name=host_details.name)
        return (True, updated_host)
    except Exception as e:
        error_message = 'Got error %s while adding initiator to host %s' % (str(e), host_details.name)
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)