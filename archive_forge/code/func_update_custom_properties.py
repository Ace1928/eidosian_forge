from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def update_custom_properties(self, attachments_service, attachment, network):
    if network.get('custom_properties'):
        current = []
        if attachment.properties:
            current = [(cp.name, str(cp.value)) for cp in attachment.properties]
        passed = [(cp.get('name'), str(cp.get('value'))) for cp in network.get('custom_properties') if cp]
        if sorted(current) != sorted(passed):
            attachment.properties = [otypes.Property(name=prop.get('name'), value=prop.get('value')) for prop in network.get('custom_properties')]
            if not self._module.check_mode:
                attachments_service.service(attachment.id).update(attachment)
            self.changed = True