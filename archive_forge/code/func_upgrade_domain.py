import datetime
import json
from copy import deepcopy
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.opensearch import compare_domain_versions
from ansible_collections.community.aws.plugins.module_utils.opensearch import ensure_tags
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_config
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_status
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_target_increment_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import normalize_opensearch
from ansible_collections.community.aws.plugins.module_utils.opensearch import parse_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import wait_for_domain_status
def upgrade_domain(client, module, source_version, target_engine_version):
    domain_name = module.params.get('domain_name')
    next_version = target_engine_version
    perform_check_only = False
    if module.check_mode:
        perform_check_only = True
    current_version = source_version
    while current_version != target_engine_version:
        v = get_target_increment_version(client, module, domain_name, target_engine_version)
        if v is None:
            next_version = target_engine_version
        if next_version != target_engine_version:
            if not module.params.get('allow_intermediate_upgrades'):
                module.fail_json(msg=f'Cannot upgrade from {source_version} to version {target_engine_version}. The highest compatible version is {next_version}')
        parameters = {'DomainName': domain_name, 'TargetVersion': next_version, 'PerformCheckOnly': perform_check_only}
        if not module.check_mode:
            wait_for_domain_status(client, module, domain_name, 'domain_available')
        try:
            client.upgrade_domain(**parameters)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg=f"Couldn't upgrade domain {domain_name} from {current_version} to {next_version}")
        if module.check_mode:
            module.exit_json(changed=True, msg=f'Would have upgraded domain from {current_version} to {next_version} if not in check mode')
        current_version = next_version
    if module.params.get('wait'):
        wait_for_domain_status(client, module, domain_name, 'domain_available')