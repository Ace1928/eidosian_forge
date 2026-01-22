from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def _filter_certificate(cert):
    if domain_name and cert['DomainName'] != domain_name:
        return False
    if arn and cert['CertificateArn'] != arn:
        return False
    return True