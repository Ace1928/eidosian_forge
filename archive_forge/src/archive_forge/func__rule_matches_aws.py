import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def _rule_matches_aws(self):
    """Checks if the local rule data matches AWS"""
    aws_rule_data = self.rule.describe()
    return all((getattr(self.rule, field) == aws_rule_data.get(field, None) for field in self.RULE_FIELDS))