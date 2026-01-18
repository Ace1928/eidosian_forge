from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def update_security_policy(self, spec, results):
    """
        Update the security policy according to the parameters
        Args:
            spec: The vSwitch spec
            results: The results dict

        Returns: True if changes have been made, else false
        """
    if not self.params['security'] or not spec.policy.security:
        return False
    security_policy = spec.policy.security
    changed = False
    sec_promiscuous_mode = self.params['security'].get('promiscuous_mode')
    sec_forged_transmits = self.params['security'].get('forged_transmits')
    sec_mac_changes = self.params['security'].get('mac_changes')
    if sec_promiscuous_mode is not None:
        results['sec_promiscuous_mode'] = sec_promiscuous_mode
        if security_policy.allowPromiscuous is not sec_promiscuous_mode:
            results['sec_promiscuous_mode_previous'] = security_policy.allowPromiscuous
            security_policy.allowPromiscuous = sec_promiscuous_mode
            changed = True
    if sec_mac_changes is not None:
        results['sec_mac_changes'] = sec_mac_changes
        if security_policy.macChanges is not sec_mac_changes:
            results['sec_mac_changes_previous'] = security_policy.macChanges
            security_policy.macChanges = sec_mac_changes
            changed = True
    if sec_forged_transmits is not None:
        results['sec_forged_transmits'] = sec_forged_transmits
        if security_policy.forgedTransmits is not sec_forged_transmits:
            results['sec_forged_transmits_previous'] = security_policy.forgedTransmits
            security_policy.forgedTransmits = sec_forged_transmits
            changed = True
    return changed