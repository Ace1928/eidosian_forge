import os
import uuid
from ansible.module_utils._text import to_bytes
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def update_key_pair_by_key_material(check_mode, ec2_client, name, key, key_material, tag_spec):
    if check_mode:
        return {'changed': True, 'key': None, 'msg': 'key pair updated'}
    new_fingerprint = get_key_fingerprint(check_mode, ec2_client, key_material)
    changed = False
    msg = 'key pair already exists'
    if key['KeyFingerprint'] != new_fingerprint:
        delete_key_pair(check_mode, ec2_client, name, finish_task=False)
        key = _import_key_pair(ec2_client, name, key_material, tag_spec)
        msg = 'key pair updated'
        changed = True
    key_data = extract_key_data(key)
    return {'changed': changed, 'key': key_data, 'msg': msg}