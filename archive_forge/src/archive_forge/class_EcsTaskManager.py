from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EcsTaskManager:
    """Handles ECS Tasks"""

    def __init__(self, module):
        self.module = module
        self.ecs = module.client('ecs', AWSRetry.jittered_backoff())

    def describe_task(self, task_name):
        try:
            response = self.ecs.describe_task_definition(aws_retry=True, taskDefinition=task_name)
            return response['taskDefinition']
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            return None

    def register_task(self, family, task_role_arn, execution_role_arn, network_mode, container_definitions, volumes, launch_type, cpu, memory, placement_constraints, runtime_platform):
        validated_containers = []
        for container in container_definitions:
            for param in ('memory', 'cpu', 'memoryReservation', 'startTimeout', 'stopTimeout'):
                if param in container:
                    container[param] = int(container[param])
            if 'portMappings' in container:
                for port_mapping in container['portMappings']:
                    for port in ('hostPort', 'containerPort'):
                        if port in port_mapping:
                            port_mapping[port] = int(port_mapping[port])
                    if network_mode == 'awsvpc' and 'hostPort' in port_mapping:
                        if port_mapping['hostPort'] != port_mapping.get('containerPort'):
                            self.module.fail_json(msg='In awsvpc network mode, host port must be set to the same as container port or not be set')
            if 'linuxParameters' in container:
                for linux_param in container.get('linuxParameters'):
                    if linux_param == 'tmpfs':
                        for tmpfs_param in container['linuxParameters']['tmpfs']:
                            if 'size' in tmpfs_param:
                                tmpfs_param['size'] = int(tmpfs_param['size'])
                    for param in ('maxSwap', 'swappiness', 'sharedMemorySize'):
                        if param in linux_param:
                            container['linuxParameters'][param] = int(container['linuxParameters'][param])
            if 'ulimits' in container:
                for limits_mapping in container['ulimits']:
                    for limit in ('softLimit', 'hardLimit'):
                        if limit in limits_mapping:
                            limits_mapping[limit] = int(limits_mapping[limit])
            validated_containers.append(container)
        params = dict(family=family, taskRoleArn=task_role_arn, containerDefinitions=container_definitions, volumes=volumes)
        if network_mode != 'default':
            params['networkMode'] = network_mode
        if cpu:
            params['cpu'] = cpu
        if memory:
            params['memory'] = memory
        if launch_type:
            params['requiresCompatibilities'] = [launch_type]
        if execution_role_arn:
            params['executionRoleArn'] = execution_role_arn
        if placement_constraints:
            params['placementConstraints'] = placement_constraints
        if runtime_platform:
            params['runtimePlatform'] = runtime_platform
        try:
            response = self.ecs.register_task_definition(aws_retry=True, **params)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Failed to register task')
        return response['taskDefinition']

    def describe_task_definitions(self, family):
        data = {'taskDefinitionArns': [], 'nextToken': None}

        def fetch():
            params = {'familyPrefix': family}
            if data['nextToken']:
                params['nextToken'] = data['nextToken']
            result = self.ecs.list_task_definitions(**params)
            data['taskDefinitionArns'] += result['taskDefinitionArns']
            data['nextToken'] = result.get('nextToken', None)
            return data['nextToken'] is not None
        while fetch():
            pass
        return list(sorted([self.ecs.describe_task_definition(taskDefinition=arn)['taskDefinition'] for arn in data['taskDefinitionArns']], key=lambda td: td['revision']))

    def deregister_task(self, taskArn):
        response = self.ecs.deregister_task_definition(taskDefinition=taskArn)
        return response['taskDefinition']