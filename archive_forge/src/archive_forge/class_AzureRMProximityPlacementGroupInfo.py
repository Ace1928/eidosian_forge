from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMProximityPlacementGroupInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMProximityPlacementGroupInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        results = []
        if self.name is not None:
            results = self.get_item()
        elif self.resource_group:
            results = self.list_resource_group()
        else:
            results = self.list_items()
        self.results['proximityplacementgroups'] = self.curated_items(results)
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.compute_client.proximity_placement_groups.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            results = [item]
        return results

    def list_resource_group(self):
        self.log('List all proximity placement groups for resource group - {0}'.format(self.resource_group))
        try:
            response = self.compute_client.proximity_placement_groups.list_by_resource_group(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def list_items(self):
        self.log('List all proximity placement groups for a subscription ')
        try:
            response = self.compute_client.proximity_placement_groups.list_by_subscription()
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def curated_items(self, raws):
        return [self.ppg_to_dict(item) for item in raws] if raws else []

    def ppg_to_dict(self, ppg):
        result = dict(id=ppg.id, name=ppg.name, location=ppg.location, tags=ppg.tags, proximity_placement_group_type=ppg.proximity_placement_group_type, virtual_machines=[dict(id=x.id) for x in ppg.virtual_machines], virtual_machine_scale_sets=[dict(id=x.id) for x in ppg.virtual_machine_scale_sets], availability_sets=[dict(id=x.id) for x in ppg.availability_sets])
        return result