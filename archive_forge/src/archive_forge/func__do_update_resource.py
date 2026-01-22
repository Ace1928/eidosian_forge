from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
def _do_update_resource(self):
    if self._preupdate_resource.get('State', None) == 'pending':
        self._wait_for_creation()
    elif self._preupdate_resource.get('State', None) == 'deleting':
        self.module.fail_json(msg='Deletion in progress, unable to update', route_tables=[self.original_resource])
    updates = self._filter_immutable_resource_attributes(self._resource_updates)
    subnets_to_add = self._subnet_updates.get('add', [])
    subnets_to_remove = self._subnet_updates.get('remove', [])
    if subnets_to_add:
        updates['AddSubnetIds'] = subnets_to_add
    if subnets_to_remove:
        updates['RemoveSubnetIds'] = subnets_to_remove
    if not updates:
        return False
    if self.module.check_mode:
        return True
    updates.update(self._get_id_params(id_list=False))
    self._modify_vpc_attachment(**updates)
    return True