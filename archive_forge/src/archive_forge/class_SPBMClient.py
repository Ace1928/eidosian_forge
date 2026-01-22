from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
class SPBMClient(SPBM):

    def __init__(self, module):
        super(SPBMClient, self).__init__(module)

    def show_capabilities(self, capabilities):
        """
        Return property instance for given capabilities
        """
        capabilities_info = []
        for capability in capabilities:
            for constraint in capability.constraint:
                if hasattr(constraint, 'propertyInstance'):
                    for propertyInstance in constraint.propertyInstance:
                        capabilities_info.append({'id': propertyInstance.id, 'value': propertyInstance.value})
        return capabilities_info

    def get_storage_policy_info(self):
        self.get_spbm_connection()
        results = dict(changed=False, spbm_profiles=[])
        profile_manager = self.spbm_content.profileManager
        profile_ids = profile_manager.PbmQueryProfile(resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), profileCategory='REQUIREMENT')
        profiles = []
        if profile_ids:
            profiles = profile_manager.PbmRetrieveContent(profileIds=profile_ids)
        for profile in profiles:
            temp_profile_info = {'name': profile.name, 'id': profile.profileId.uniqueId, 'description': profile.description, 'constraints_sub_profiles': []}
            if hasattr(profile.constraints, 'subProfiles'):
                subprofiles = profile.constraints.subProfiles
                temp_sub_profiles = []
                for subprofile in subprofiles:
                    rule_set_info = self.show_capabilities(subprofile.capability)
                    for _rule_set_info in rule_set_info:
                        if isinstance(_rule_set_info['value'], pbm.capability.types.DiscreteSet):
                            _rule_set_info['value'] = _rule_set_info['value'].values
                    temp_sub_profiles.append({'rule_set_name': subprofile.name, 'rule_set_info': rule_set_info})
                temp_profile_info['constraints_sub_profiles'] = temp_sub_profiles
            results['spbm_profiles'].append(temp_profile_info)
        self.module.exit_json(**results)