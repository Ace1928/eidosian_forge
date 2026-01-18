import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_feedback_forwarding(connection, module, identity, identity_notifications):
    if identity_notifications is None:
        current = True
    elif 'ForwardingEnabled' in identity_notifications:
        current = identity_notifications['ForwardingEnabled']
    else:
        current = False
    required = module.params.get('feedback_forwarding')
    if current != required:
        try:
            if not module.check_mode:
                connection.set_identity_feedback_forwarding_enabled(Identity=identity, ForwardingEnabled=required, aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to set identity feedback forwarding for {identity}')
        return True
    return False