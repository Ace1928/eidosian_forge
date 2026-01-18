import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_identity_notifications(connection, module):
    identity = module.params.get('identity')
    changed = False
    identity_notifications = get_identity_notifications(connection, module, identity)
    for notification_type in ('Bounce', 'Complaint', 'Delivery'):
        changed |= update_notification_topic(connection, module, identity, identity_notifications, notification_type)
        changed |= update_notification_topic_headers(connection, module, identity, identity_notifications, notification_type)
    changed |= update_feedback_forwarding(connection, module, identity, identity_notifications)
    if changed or identity_notifications is None:
        if module.check_mode:
            identity_notifications = create_mock_notifications_response(module)
        else:
            identity_notifications = get_identity_notifications(connection, module, identity, retries=4)
    return (changed, identity_notifications)