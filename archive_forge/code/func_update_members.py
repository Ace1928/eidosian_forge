from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def update_members(self, group_id, client):
    current_members = []
    if self.present_members or self.absent_members:
        current_members = [object.object_id for object in list(client.groups.get_group_members(group_id))]
    if self.present_members:
        present_members_by_object_id = self.dictionary_from_object_urls(self.present_members)
        members_to_add = list(set(present_members_by_object_id.keys()) - set(current_members))
        if members_to_add:
            for member_object_id in members_to_add:
                client.groups.add_member(group_id, present_members_by_object_id[member_object_id])
            self.results['changed'] = True
    if self.absent_members:
        members_to_remove = list(set(self.absent_members).intersection(set(current_members)))
        if members_to_remove:
            for member in members_to_remove:
                client.groups.remove_member(group_id, member)
            self.results['changed'] = True