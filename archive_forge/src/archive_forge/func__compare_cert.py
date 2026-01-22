from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _compare_cert(cert_a, cert_b):
    if not cert_a and (not cert_b):
        return True
    if not cert_a or not cert_b:
        return False
    cert_a.replace('\r', '')
    cert_a.replace('\n', '')
    cert_a.replace(' ', '')
    cert_b.replace('\r', '')
    cert_b.replace('\n', '')
    cert_b.replace(' ', '')
    return cert_a == cert_b