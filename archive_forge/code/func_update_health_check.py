import uuid
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def update_health_check(existing_check):
    changes = dict()
    existing_config = existing_check.get('HealthCheckConfig')
    check_id = existing_check.get('Id')
    resource_path = module.params.get('resource_path', None)
    if resource_path and resource_path != existing_config.get('ResourcePath'):
        changes['ResourcePath'] = resource_path
    search_string = module.params.get('string_match', None)
    if search_string and search_string != existing_config.get('SearchString'):
        changes['SearchString'] = search_string
    type_in = module.params.get('type', None)
    if type_in != 'CALCULATED':
        failure_threshold = module.params.get('failure_threshold', None)
        if failure_threshold and failure_threshold != existing_config.get('FailureThreshold'):
            changes['FailureThreshold'] = failure_threshold
    disabled = module.params.get('disabled', None)
    if disabled is not None and disabled != existing_config.get('Disabled'):
        changes['Disabled'] = module.params.get('disabled')
    if module.params.get('health_check_id') or module.params.get('use_unique_names'):
        ip_address = module.params.get('ip_address', None)
        if ip_address is not None and ip_address != existing_config.get('IPAddress'):
            changes['IPAddress'] = module.params.get('ip_address')
        port = module.params.get('port', None)
        if port is not None and port != existing_config.get('Port'):
            changes['Port'] = module.params.get('port')
        fqdn = module.params.get('fqdn', None)
        if fqdn is not None and fqdn != existing_config.get('FullyQualifiedDomainName'):
            changes['FullyQualifiedDomainName'] = module.params.get('fqdn')
        if type_in == 'CALCULATED':
            child_health_checks = module.params.get('child_health_checks', None)
            if child_health_checks is not None and child_health_checks != existing_config.get('ChildHealthChecks'):
                changes['ChildHealthChecks'] = module.params.get('child_health_checks')
            health_threshold = module.params.get('health_threshold', None)
            if health_threshold is not None and health_threshold != existing_config.get('HealthThreshold'):
                changes['HealthThreshold'] = module.params.get('health_threshold')
    if not changes:
        return (False, None, check_id)
    if module.check_mode:
        return (True, 'update', check_id)
    version_id = existing_check.get('HealthCheckVersion', 1)
    try:
        client.update_health_check(HealthCheckId=check_id, HealthCheckVersion=version_id, **changes)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to update health check.', id=check_id)
    return (True, 'update', check_id)