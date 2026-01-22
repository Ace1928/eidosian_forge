from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class TargetInfoGatherer(object):

    def __init__(self, module, instance_id, get_unused_target_groups):
        self.module = module
        try:
            self.ec2 = self.module.client('ec2', retry_decorator=AWSRetry.jittered_backoff(retries=10))
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg="Couldn't connect to ec2")
        try:
            self.elbv2 = self.module.client('elbv2', retry_decorator=AWSRetry.jittered_backoff(retries=10))
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Could not connect to elbv2')
        self.instance_id = instance_id
        self.get_unused_target_groups = get_unused_target_groups
        self.tgs = self._get_target_groups()

    def _get_instance_ips(self):
        """Fetch all IPs associated with this instance so that we can determine
        whether or not an instance is in an IP-based target group"""
        try:
            reservations = self.ec2.describe_instances(InstanceIds=[self.instance_id], aws_retry=True)['Reservations']
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg=f"Could not get instance info for instance '{self.instance_id}'")
        if len(reservations) < 1:
            self.module.fail_json(msg=f'Instance ID {self.instance_id} could not be found')
        instance = reservations[0]['Instances'][0]
        ips = set()
        ips.add(instance['PrivateIpAddress'])
        for nic in instance['NetworkInterfaces']:
            ips.add(nic['PrivateIpAddress'])
            for ip in nic['PrivateIpAddresses']:
                ips.add(ip['PrivateIpAddress'])
        return list(ips)

    def _get_target_group_objects(self):
        """helper function to build a list of TargetGroup objects based on
        the AWS API"""
        try:
            paginator = self.elbv2.get_paginator('describe_target_groups')
            tg_response = paginator.paginate().build_full_result()
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Could not describe target groups')
        target_groups = []
        for each_tg in tg_response['TargetGroups']:
            if not self.get_unused_target_groups and len(each_tg['LoadBalancerArns']) < 1:
                continue
            target_groups.append(TargetGroup(target_group_arn=each_tg['TargetGroupArn'], target_group_type=each_tg['TargetType']))
        return target_groups

    def _get_target_descriptions(self, target_groups):
        """Helper function to build a list of all the target descriptions
        for this target in a target group"""
        tgs = set()
        for tg in target_groups:
            try:
                response = self.elbv2.describe_target_health(TargetGroupArn=tg.target_group_arn, aws_retry=True)
            except (BotoCoreError, ClientError) as e:
                self.module.fail_json_aws(e, msg='Could not describe target ' + f'health for target group {tg.target_group_arn}')
            for t in response['TargetHealthDescriptions']:
                if t['Target']['Id'] == self.instance_id or t['Target']['Id'] in self.instance_ips:
                    az = t['Target']['AvailabilityZone'] if 'AvailabilityZone' in t['Target'] else None
                    tg.add_target(t['Target']['Id'], t['Target']['Port'], az, t['TargetHealth'])
                    tgs.add(tg)
        return list(tgs)

    def _get_target_groups(self):
        self.instance_ips = self._get_instance_ips()
        target_groups = self._get_target_group_objects()
        return self._get_target_descriptions(target_groups)