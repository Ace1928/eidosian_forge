import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EcsEcr:

    def __init__(self, module):
        self.ecr = module.client('ecr')
        self.sts = module.client('sts')
        self.check_mode = module.check_mode
        self.changed = False
        self.skipped = False

    def get_repository(self, registry_id, name):
        try:
            res = self.ecr.describe_repositories(repositoryNames=[name], **build_kwargs(registry_id))
            repos = res.get('repositories')
            return repos and repos[0]
        except is_boto3_error_code('RepositoryNotFoundException'):
            return None

    def get_repository_policy(self, registry_id, name):
        try:
            res = self.ecr.get_repository_policy(repositoryName=name, **build_kwargs(registry_id))
            text = res.get('policyText')
            return text and json.loads(text)
        except is_boto3_error_code(['RepositoryNotFoundException', 'RepositoryPolicyNotFoundException']):
            return None

    def create_repository(self, registry_id, name, image_tag_mutability, encryption_configuration):
        if registry_id:
            default_registry_id = self.sts.get_caller_identity().get('Account')
            if registry_id != default_registry_id:
                raise Exception(f'Cannot create repository in registry {registry_id}.  Would be created in {default_registry_id} instead.')
        if encryption_configuration is None:
            encryption_configuration = dict(encryptionType='AES256')
        if not self.check_mode:
            repo = self.ecr.create_repository(repositoryName=name, imageTagMutability=image_tag_mutability, encryptionConfiguration=encryption_configuration).get('repository')
            self.changed = True
            return repo
        else:
            self.skipped = True
            return dict(repositoryName=name)

    def set_repository_policy(self, registry_id, name, policy_text, force):
        if not self.check_mode:
            policy = self.ecr.set_repository_policy(repositoryName=name, policyText=policy_text, force=force, **build_kwargs(registry_id))
            self.changed = True
            return policy
        else:
            self.skipped = True
            if self.get_repository(registry_id, name) is None:
                printable = name
                if registry_id:
                    printable = f'{registry_id}:{name}'
                raise Exception(f'could not find repository {printable}')
            return

    def delete_repository(self, registry_id, name, force):
        if not self.check_mode:
            repo = self.ecr.delete_repository(repositoryName=name, force=force, **build_kwargs(registry_id))
            self.changed = True
            return repo
        else:
            repo = self.get_repository(registry_id, name)
            if repo:
                self.skipped = True
                return repo
            return None

    def delete_repository_policy(self, registry_id, name):
        if not self.check_mode:
            policy = self.ecr.delete_repository_policy(repositoryName=name, **build_kwargs(registry_id))
            self.changed = True
            return policy
        else:
            policy = self.get_repository_policy(registry_id, name)
            if policy:
                self.skipped = True
                return policy
            return None

    def put_image_tag_mutability(self, registry_id, name, new_mutability_configuration):
        repo = self.get_repository(registry_id, name)
        current_mutability_configuration = repo.get('imageTagMutability')
        if current_mutability_configuration != new_mutability_configuration:
            if not self.check_mode:
                self.ecr.put_image_tag_mutability(repositoryName=name, imageTagMutability=new_mutability_configuration, **build_kwargs(registry_id))
            else:
                self.skipped = True
            self.changed = True
        repo['imageTagMutability'] = new_mutability_configuration
        return repo

    def get_lifecycle_policy(self, registry_id, name):
        try:
            res = self.ecr.get_lifecycle_policy(repositoryName=name, **build_kwargs(registry_id))
            text = res.get('lifecyclePolicyText')
            return text and json.loads(text)
        except is_boto3_error_code(['LifecyclePolicyNotFoundException', 'RepositoryNotFoundException']):
            return None

    def put_lifecycle_policy(self, registry_id, name, policy_text):
        if not self.check_mode:
            policy = self.ecr.put_lifecycle_policy(repositoryName=name, lifecyclePolicyText=policy_text, **build_kwargs(registry_id))
            self.changed = True
            return policy
        else:
            self.skipped = True
            if self.get_repository(registry_id, name) is None:
                printable = name
                if registry_id:
                    printable = f'{registry_id}:{name}'
                raise Exception(f'could not find repository {printable}')
            return

    def purge_lifecycle_policy(self, registry_id, name):
        if not self.check_mode:
            policy = self.ecr.delete_lifecycle_policy(repositoryName=name, **build_kwargs(registry_id))
            self.changed = True
            return policy
        else:
            policy = self.get_lifecycle_policy(registry_id, name)
            if policy:
                self.skipped = True
                return policy
            return None

    def put_image_scanning_configuration(self, registry_id, name, scan_on_push):
        if not self.check_mode:
            if registry_id:
                scan = self.ecr.put_image_scanning_configuration(registryId=registry_id, repositoryName=name, imageScanningConfiguration={'scanOnPush': scan_on_push})
            else:
                scan = self.ecr.put_image_scanning_configuration(repositoryName=name, imageScanningConfiguration={'scanOnPush': scan_on_push})
            self.changed = True
            return scan
        else:
            self.skipped = True
            return None