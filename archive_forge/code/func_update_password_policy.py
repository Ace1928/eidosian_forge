from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def update_password_policy(self, module, policy):
    min_pw_length = module.params.get('min_pw_length')
    require_symbols = module.params.get('require_symbols')
    require_numbers = module.params.get('require_numbers')
    require_uppercase = module.params.get('require_uppercase')
    require_lowercase = module.params.get('require_lowercase')
    allow_pw_change = module.params.get('allow_pw_change')
    pw_max_age = module.params.get('pw_max_age')
    pw_reuse_prevent = module.params.get('pw_reuse_prevent')
    pw_expire = module.params.get('pw_expire')
    update_parameters = dict(MinimumPasswordLength=min_pw_length, RequireSymbols=require_symbols, RequireNumbers=require_numbers, RequireUppercaseCharacters=require_uppercase, RequireLowercaseCharacters=require_lowercase, AllowUsersToChangePassword=allow_pw_change, HardExpiry=pw_expire)
    if pw_reuse_prevent:
        update_parameters.update(PasswordReusePrevention=pw_reuse_prevent)
    if pw_max_age:
        update_parameters.update(MaxPasswordAge=pw_max_age)
    try:
        original_policy = self.policy_to_dict(policy)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        original_policy = {}
    try:
        results = policy.update(**update_parameters)
        policy.reload()
        updated_policy = self.policy_to_dict(policy)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg="Couldn't update IAM Password Policy")
    changed = original_policy != updated_policy
    return (changed, updated_policy, camel_dict_to_snake_dict(results))