from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_affinity_label_mappings(module):
    affinityLabelMappings = list()
    for affinityLabelMapping in module.params['affinity_label_mappings']:
        affinityLabelMappings.append(otypes.RegistrationAffinityLabelMapping(from_=otypes.AffinityLabel(name=affinityLabelMapping['source_name']) if affinityLabelMapping['source_name'] else None, to=otypes.AffinityLabel(name=affinityLabelMapping['dest_name']) if affinityLabelMapping['dest_name'] else None))
    return affinityLabelMappings