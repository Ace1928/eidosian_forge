from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@staticmethod
def _read_gateway_tape_response(tapes, aws_response):
    for tape in aws_response['TapeInfos']:
        tape_obj = camel_dict_to_snake_dict(tape)
        if 'gateway_arn' in tape_obj:
            del tape_obj['gateway_arn']
        tapes.append(tape_obj)
    return aws_response['Marker'] if 'Marker' in aws_response else None