from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_ecs_arn(self):
    try:
        ecs_instances_arns = self.ecs.list_container_instances(cluster=self.cluster)['containerInstanceArns']
        ec2_instances = self.ecs.describe_container_instances(cluster=self.cluster, containerInstances=ecs_instances_arns)['containerInstances']
    except (ClientError, EndpointConnectionError) as e:
        self.module.fail_json(msg=f"Can't connect to the cluster - {str(e)}")
    try:
        ecs_arn = next((inst for inst in ec2_instances if inst['ec2InstanceId'] == self.ec2_id))['containerInstanceArn']
    except StopIteration:
        self.module.fail_json(msg=f'EC2 instance Id not found in ECS cluster - {str(self.cluster)}')
    return ecs_arn