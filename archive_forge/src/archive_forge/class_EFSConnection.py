from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EFSConnection(object):
    DEFAULT_WAIT_TIMEOUT_SECONDS = 0
    STATE_CREATING = 'creating'
    STATE_AVAILABLE = 'available'
    STATE_DELETING = 'deleting'
    STATE_DELETED = 'deleted'

    def __init__(self, module):
        self.connection = module.client('efs')
        region = module.region
        self.module = module
        self.region = region
        self.wait = module.params.get('wait')
        self.wait_timeout = module.params.get('wait_timeout')

    def get_file_systems(self, **kwargs):
        """
        Returns generator of file systems including all attributes of FS
        """
        items = iterate_all('FileSystems', self.connection.describe_file_systems, **kwargs)
        for item in items:
            item['Name'] = item['CreationToken']
            item['CreationTime'] = str(item['CreationTime'])
            '\n            In the time when MountPoint was introduced there was a need to add a suffix of network path before one could use it\n            AWS updated it and now there is no need to add a suffix. MountPoint is left for back-compatibility purpose\n            And new FilesystemAddress variable is introduced for direct use with other modules (e.g. mount)\n            AWS documentation is available here:\n            https://docs.aws.amazon.com/efs/latest/ug/gs-step-three-connect-to-ec2-instance.html\n            '
            item['MountPoint'] = f'.{item['FileSystemId']}.efs.{self.region}.amazonaws.com:/'
            item['FilesystemAddress'] = f'{item['FileSystemId']}.efs.{self.region}.amazonaws.com:/'
            if 'Timestamp' in item['SizeInBytes']:
                item['SizeInBytes']['Timestamp'] = str(item['SizeInBytes']['Timestamp'])
            if item['LifeCycleState'] == self.STATE_AVAILABLE:
                item['Tags'] = self.get_tags(FileSystemId=item['FileSystemId'])
                item['MountTargets'] = list(self.get_mount_targets(FileSystemId=item['FileSystemId']))
            else:
                item['Tags'] = {}
                item['MountTargets'] = []
            yield item

    def get_tags(self, **kwargs):
        """
        Returns tag list for selected instance of EFS
        """
        tags = self.connection.describe_tags(**kwargs)['Tags']
        return tags

    def get_mount_targets(self, **kwargs):
        """
        Returns mount targets for selected instance of EFS
        """
        targets = iterate_all('MountTargets', self.connection.describe_mount_targets, **kwargs)
        for target in targets:
            if target['LifeCycleState'] == self.STATE_AVAILABLE:
                target['SecurityGroups'] = list(self.get_security_groups(MountTargetId=target['MountTargetId']))
            else:
                target['SecurityGroups'] = []
            yield target

    def get_security_groups(self, **kwargs):
        """
        Returns security groups for selected instance of EFS
        """
        return iterate_all('SecurityGroups', self.connection.describe_mount_target_security_groups, **kwargs)

    def get_file_system_id(self, name):
        """
        Returns ID of instance by instance name
        """
        info = first_or_default(iterate_all('FileSystems', self.connection.describe_file_systems, CreationToken=name))
        return info and info['FileSystemId'] or None

    def get_file_system_state(self, name, file_system_id=None):
        """
        Returns state of filesystem by EFS id/name
        """
        info = first_or_default(iterate_all('FileSystems', self.connection.describe_file_systems, CreationToken=name, FileSystemId=file_system_id))
        return info and info['LifeCycleState'] or self.STATE_DELETED

    def get_mount_targets_in_state(self, file_system_id, states=None):
        """
        Returns states of mount targets of selected EFS with selected state(s) (optional)
        """
        targets = iterate_all('MountTargets', self.connection.describe_mount_targets, FileSystemId=file_system_id)
        if states:
            if not isinstance(states, list):
                states = [states]
            targets = filter(lambda target: target['LifeCycleState'] in states, targets)
        return list(targets)

    def get_throughput_mode(self, **kwargs):
        """
        Returns throughput mode for selected EFS instance
        """
        info = first_or_default(iterate_all('FileSystems', self.connection.describe_file_systems, **kwargs))
        return info and info['ThroughputMode'] or None

    def get_provisioned_throughput_in_mibps(self, **kwargs):
        """
        Returns throughput mode for selected EFS instance
        """
        info = first_or_default(iterate_all('FileSystems', self.connection.describe_file_systems, **kwargs))
        return info.get('ProvisionedThroughputInMibps', None)

    def create_file_system(self, name, performance_mode, encrypt, kms_key_id, throughput_mode, provisioned_throughput_in_mibps):
        """
        Creates new filesystem with selected name
        """
        changed = False
        state = self.get_file_system_state(name)
        params = {}
        params['CreationToken'] = name
        params['PerformanceMode'] = performance_mode
        if encrypt:
            params['Encrypted'] = encrypt
        if kms_key_id is not None:
            params['KmsKeyId'] = kms_key_id
        if throughput_mode:
            params['ThroughputMode'] = throughput_mode
        if provisioned_throughput_in_mibps:
            params['ProvisionedThroughputInMibps'] = provisioned_throughput_in_mibps
        if state in [self.STATE_DELETING, self.STATE_DELETED]:
            wait_for(lambda: self.get_file_system_state(name), self.STATE_DELETED)
            try:
                self.connection.create_file_system(**params)
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Unable to create file system.')
        wait_for(lambda: self.get_file_system_state(name), self.STATE_AVAILABLE, self.wait_timeout)
        return changed

    def update_file_system(self, name, throughput_mode, provisioned_throughput_in_mibps):
        """
        Update filesystem with new throughput settings
        """
        changed = False
        state = self.get_file_system_state(name)
        if state in [self.STATE_AVAILABLE, self.STATE_CREATING]:
            fs_id = self.get_file_system_id(name)
            current_mode = self.get_throughput_mode(FileSystemId=fs_id)
            current_throughput = self.get_provisioned_throughput_in_mibps(FileSystemId=fs_id)
            params = dict()
            if throughput_mode and throughput_mode != current_mode:
                params['ThroughputMode'] = throughput_mode
            if provisioned_throughput_in_mibps and provisioned_throughput_in_mibps != current_throughput:
                params['ProvisionedThroughputInMibps'] = provisioned_throughput_in_mibps
            if len(params) > 0:
                wait_for(lambda: self.get_file_system_state(name), self.STATE_AVAILABLE, self.wait_timeout)
                try:
                    self.connection.update_file_system(FileSystemId=fs_id, **params)
                    changed = True
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg='Unable to update file system.')
        return changed

    def update_lifecycle_policy(self, name, transition_to_ia):
        """
        Update filesystem with new lifecycle policy.
        """
        changed = False
        state = self.get_file_system_state(name)
        if state in [self.STATE_AVAILABLE, self.STATE_CREATING]:
            fs_id = self.get_file_system_id(name)
            current_policies = self.connection.describe_lifecycle_configuration(FileSystemId=fs_id)
            if transition_to_ia == 'None':
                LifecyclePolicies = []
            else:
                LifecyclePolicies = [{'TransitionToIA': 'AFTER_' + transition_to_ia + '_DAYS'}]
            if current_policies.get('LifecyclePolicies') != LifecyclePolicies:
                response = self.connection.put_lifecycle_configuration(FileSystemId=fs_id, LifecyclePolicies=LifecyclePolicies)
                changed = True
        return changed

    def converge_file_system(self, name, tags, purge_tags, targets, throughput_mode, provisioned_throughput_in_mibps):
        """
        Change attributes (mount targets and tags) of filesystem by name
        """
        result = False
        fs_id = self.get_file_system_id(name)
        if tags is not None:
            tags_need_modify, tags_to_delete = compare_aws_tags(boto3_tag_list_to_ansible_dict(self.get_tags(FileSystemId=fs_id)), tags, purge_tags)
            if tags_to_delete:
                try:
                    self.connection.delete_tags(FileSystemId=fs_id, TagKeys=tags_to_delete)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg='Unable to delete tags.')
                result = True
            if tags_need_modify:
                try:
                    self.connection.create_tags(FileSystemId=fs_id, Tags=ansible_dict_to_boto3_tag_list(tags_need_modify))
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg='Unable to create tags.')
                result = True
        if targets is not None:
            incomplete_states = [self.STATE_CREATING, self.STATE_DELETING]
            wait_for(lambda: len(self.get_mount_targets_in_state(fs_id, incomplete_states)), 0)
            current_targets = _index_by_key('SubnetId', self.get_mount_targets(FileSystemId=fs_id))
            targets = _index_by_key('SubnetId', targets)
            targets_to_create, intersection, targets_to_delete = dict_diff(current_targets, targets, True)
            changed = [sid for sid in intersection if not targets_equal(['SubnetId', 'IpAddress', 'NetworkInterfaceId'], current_targets[sid], targets[sid])]
            targets_to_delete = list(targets_to_delete) + changed
            targets_to_create = list(targets_to_create) + changed
            if targets_to_delete:
                for sid in targets_to_delete:
                    self.connection.delete_mount_target(MountTargetId=current_targets[sid]['MountTargetId'])
                wait_for(lambda: len(self.get_mount_targets_in_state(fs_id, incomplete_states)), 0)
                result = True
            if targets_to_create:
                for sid in targets_to_create:
                    self.connection.create_mount_target(FileSystemId=fs_id, **targets[sid])
                wait_for(lambda: len(self.get_mount_targets_in_state(fs_id, incomplete_states)), 0, self.wait_timeout)
                result = True
            security_groups_to_update = [sid for sid in intersection if 'SecurityGroups' in targets[sid] and current_targets[sid]['SecurityGroups'] != targets[sid]['SecurityGroups']]
            if security_groups_to_update:
                for sid in security_groups_to_update:
                    self.connection.modify_mount_target_security_groups(MountTargetId=current_targets[sid]['MountTargetId'], SecurityGroups=targets[sid].get('SecurityGroups', None))
                result = True
        return result

    def delete_file_system(self, name, file_system_id=None):
        """
        Removes EFS instance by id/name
        """
        result = False
        state = self.get_file_system_state(name, file_system_id)
        if state in [self.STATE_CREATING, self.STATE_AVAILABLE]:
            wait_for(lambda: self.get_file_system_state(name), self.STATE_AVAILABLE)
            if not file_system_id:
                file_system_id = self.get_file_system_id(name)
            self.delete_mount_targets(file_system_id)
            self.connection.delete_file_system(FileSystemId=file_system_id)
            result = True
        if self.wait:
            wait_for(lambda: self.get_file_system_state(name), self.STATE_DELETED, self.wait_timeout)
        return result

    def delete_mount_targets(self, file_system_id):
        """
        Removes mount targets by EFS id
        """
        wait_for(lambda: len(self.get_mount_targets_in_state(file_system_id, self.STATE_CREATING)), 0)
        targets = self.get_mount_targets_in_state(file_system_id, self.STATE_AVAILABLE)
        for target in targets:
            self.connection.delete_mount_target(MountTargetId=target['MountTargetId'])
        wait_for(lambda: len(self.get_mount_targets_in_state(file_system_id, self.STATE_DELETING)), 0)
        return len(targets) > 0