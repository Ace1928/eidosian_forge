from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
class AzureRMProximityPlacementGroup(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.tags = None
        super(AzureRMProximityPlacementGroup, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        proximity_placement_group = None
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        try:
            self.log('Fetching Proximity placement group {0}'.format(self.name))
            proximity_placement_group = self.compute_client.proximity_placement_groups.get(self.resource_group, self.name)
            results = self.ppg_to_dict(proximity_placement_group)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                self.tags = results['tags']
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                proximity_placement_group_new = self.compute_models.ProximityPlacementGroup(location=self.location, proximity_placement_group_type='Standard')
                if self.tags:
                    proximity_placement_group_new.tags = self.tags
                self.results['state'] = self.create_or_update_placementgroup(proximity_placement_group_new)
            elif self.state == 'absent':
                self.delete_placementgroup()
                self.results['state'] = 'Deleted'
        return self.results

    def create_or_update_placementgroup(self, proximity_placement_group):
        try:
            response = self.compute_client.proximity_placement_groups.create_or_update(resource_group_name=self.resource_group, proximity_placement_group_name=self.name, parameters=proximity_placement_group)
        except Exception as exc:
            self.fail('Error creating or updating proximity placement group {0} - {1}'.format(self.name, str(exc)))
        return self.ppg_to_dict(response)

    def delete_placementgroup(self):
        try:
            response = self.compute_client.proximity_placement_groups.delete(resource_group_name=self.resource_group, proximity_placement_group_name=self.name)
        except Exception as exc:
            self.fail('Error deleting proximity placement group {0} - {1}'.format(self.name, str(exc)))
        return response

    def ppg_to_dict(self, proximityplacementgroup):
        result = proximityplacementgroup.as_dict()
        result['tags'] = proximityplacementgroup.tags
        return result