from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def _generate_updated_resource(self):
    """
        Merges all pending changes into self.updated_resource
        Used during check mode where it's not possible to get and
        refresh the resource
        """
    return self._merge_resource_changes(filter_immutable=False)