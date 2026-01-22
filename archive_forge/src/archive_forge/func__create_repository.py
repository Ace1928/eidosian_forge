from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _create_repository(self):
    try:
        result = self._client.create_repository(repositoryName=self._module.params['name'], repositoryDescription=self._module.params['description'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self._module.fail_json_aws(e, msg="couldn't create repository")
    return result