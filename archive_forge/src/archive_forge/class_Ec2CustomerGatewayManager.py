from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class Ec2CustomerGatewayManager:

    def __init__(self, module):
        self.module = module
        try:
            self.ec2 = module.client('ec2')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to connect to AWS')

    @AWSRetry.jittered_backoff(delay=2, max_delay=30, retries=6, catch_extra_error_codes=['IncorrectState'])
    def ensure_cgw_absent(self, gw_id):
        response = self.ec2.delete_customer_gateway(DryRun=False, CustomerGatewayId=gw_id)
        return response

    def ensure_cgw_present(self, bgp_asn, ip_address):
        if not bgp_asn:
            bgp_asn = 65000
        response = self.ec2.create_customer_gateway(DryRun=False, Type='ipsec.1', PublicIp=ip_address, BgpAsn=bgp_asn)
        return response

    def tag_cgw_name(self, gw_id, name):
        response = self.ec2.create_tags(DryRun=False, Resources=[gw_id], Tags=[{'Key': 'Name', 'Value': name}])
        return response

    def describe_gateways(self, ip_address):
        response = self.ec2.describe_customer_gateways(DryRun=False, Filters=[{'Name': 'state', 'Values': ['available']}, {'Name': 'ip-address', 'Values': [ip_address]}])
        return response