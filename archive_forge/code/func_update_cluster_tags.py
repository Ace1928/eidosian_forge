import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_cluster_tags(client, module, arn):
    new_tags = module.params.get('tags')
    if new_tags is None:
        return False
    purge_tags = module.params.get('purge_tags')
    try:
        existing_tags = client.list_tags_for_resource(ResourceArn=arn, aws_retry=True)['Tags']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f"Unable to retrieve tags for cluster '{arn}'")
    tags_to_add, tags_to_remove = compare_aws_tags(existing_tags, new_tags, purge_tags=purge_tags)
    if not module.check_mode:
        try:
            if tags_to_remove:
                client.untag_resource(ResourceArn=arn, TagKeys=tags_to_remove, aws_retry=True)
            if tags_to_add:
                client.tag_resource(ResourceArn=arn, Tags=tags_to_add, aws_retry=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f"Unable to set tags for cluster '{arn}'")
    changed = bool(tags_to_add) or bool(tags_to_remove)
    return changed