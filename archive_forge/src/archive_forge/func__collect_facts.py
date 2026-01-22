from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _collect_facts(resource):
    """Transform cluster information to dict."""
    facts = {'identifier': resource['ClusterIdentifier'], 'status': resource['ClusterStatus'], 'username': resource['MasterUsername'], 'db_name': resource['DBName'], 'maintenance_window': resource['PreferredMaintenanceWindow'], 'enhanced_vpc_routing': resource['EnhancedVpcRouting']}
    for node in resource['ClusterNodes']:
        if node['NodeRole'] in ('SHARED', 'LEADER'):
            facts['private_ip_address'] = node['PrivateIPAddress']
            if facts['enhanced_vpc_routing'] is False:
                facts['public_ip_address'] = node['PublicIPAddress']
            else:
                facts['public_ip_address'] = None
            break
    facts['create_time'] = None
    facts['url'] = None
    facts['port'] = None
    facts['availability_zone'] = None
    facts['tags'] = {}
    if resource['ClusterStatus'] != 'creating':
        facts['create_time'] = resource['ClusterCreateTime']
        facts['url'] = resource['Endpoint']['Address']
        facts['port'] = resource['Endpoint']['Port']
        facts['availability_zone'] = resource['AvailabilityZone']
        facts['tags'] = boto3_tag_list_to_ansible_dict(resource['Tags'])
    return facts