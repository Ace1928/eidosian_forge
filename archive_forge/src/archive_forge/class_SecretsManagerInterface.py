import json
from traceback import format_exc
from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class SecretsManagerInterface(object):
    """An interface with SecretsManager"""

    def __init__(self, module):
        self.module = module
        self.client = self.module.client('secretsmanager')

    def get_secret(self, name):
        try:
            secret = self.client.describe_secret(SecretId=name)
        except self.client.exceptions.ResourceNotFoundException:
            secret = None
        except Exception as e:
            self.module.fail_json_aws(e, msg='Failed to describe secret')
        return secret

    def get_resource_policy(self, name):
        try:
            resource_policy = self.client.get_resource_policy(SecretId=name)
        except self.client.exceptions.ResourceNotFoundException:
            resource_policy = None
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to get secret resource policy')
        return resource_policy

    def create_secret(self, secret):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            created_secret = self.client.create_secret(**secret.create_args)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to create secret')
        if secret.rotation_enabled:
            response = self.update_rotation(secret)
            created_secret['VersionId'] = response.get('VersionId')
        return created_secret

    def update_secret(self, secret):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            response = self.client.update_secret(**secret.update_args)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to update secret')
        return response

    def put_resource_policy(self, secret):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            json.loads(secret.secret_resource_policy_args.get('ResourcePolicy'))
        except (TypeError, ValueError) as e:
            self.module.fail_json(msg=f'Failed to parse resource policy as JSON: {str(e)}', exception=format_exc())
        try:
            response = self.client.put_resource_policy(**secret.secret_resource_policy_args)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to update secret resource policy')
        return response

    def remove_replication(self, name, regions):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            replica_regions = []
            response = self.client.remove_regions_from_replication(SecretId=name, RemoveReplicaRegions=regions)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to replicate secret')
        return response

    def replicate_secret(self, name, regions):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            replica_regions = []
            for replica in regions:
                if replica['kms_key_id']:
                    replica_regions.append({'Region': replica['region'], 'KmsKeyId': replica['kms_key_id']})
                else:
                    replica_regions.append({'Region': replica['region']})
            response = self.client.replicate_secret_to_regions(SecretId=name, AddReplicaRegions=replica_regions)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to replicate secret')
        return response

    def restore_secret(self, name):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            response = self.client.restore_secret(SecretId=name)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to restore secret')
        return response

    def delete_secret(self, name, recovery_window):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            if recovery_window == 0:
                response = self.client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
            else:
                response = self.client.delete_secret(SecretId=name, RecoveryWindowInDays=recovery_window)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to delete secret')
        return response

    def delete_resource_policy(self, name):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            response = self.client.delete_resource_policy(SecretId=name)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to delete secret resource policy')
        return response

    def update_rotation(self, secret):
        if secret.rotation_enabled:
            try:
                response = self.client.rotate_secret(SecretId=secret.name, RotationLambdaARN=secret.rotation_lambda_arn, RotationRules=secret.rotation_rules)
            except (BotoCoreError, ClientError) as e:
                self.module.fail_json_aws(e, msg='Failed to rotate secret secret')
        else:
            try:
                response = self.client.cancel_rotate_secret(SecretId=secret.name)
            except (BotoCoreError, ClientError) as e:
                self.module.fail_json_aws(e, msg='Failed to cancel rotation')
        return response

    def tag_secret(self, secret_name, tags):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            self.client.tag_resource(SecretId=secret_name, Tags=tags)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to add tag(s) to secret')

    def untag_secret(self, secret_name, tag_keys):
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        try:
            self.client.untag_resource(SecretId=secret_name, TagKeys=tag_keys)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to remove tag(s) from secret')

    def secrets_match(self, desired_secret, current_secret):
        """Compare secrets except tags and rotation

        Args:
            desired_secret: camel dict representation of the desired secret state.
            current_secret: secret reference as returned by the secretsmanager api.

        Returns: bool
        """
        if desired_secret.description != current_secret.get('Description', ''):
            return False
        if desired_secret.kms_key_id != current_secret.get('KmsKeyId'):
            return False
        current_secret_value = self.client.get_secret_value(SecretId=current_secret.get('Name'))
        if desired_secret.secret_type == 'SecretBinary':
            desired_value = to_bytes(desired_secret.secret)
        else:
            desired_value = desired_secret.secret
        if desired_value != current_secret_value.get(desired_secret.secret_type):
            return False
        return True