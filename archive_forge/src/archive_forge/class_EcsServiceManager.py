from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EcsServiceManager:
    """Handles ECS Services"""

    def __init__(self, module):
        self.module = module
        self.ecs = module.client('ecs')

    @AWSRetry.jittered_backoff(retries=5, delay=5, backoff=2.0)
    def list_services_with_backoff(self, **kwargs):
        paginator = self.ecs.get_paginator('list_services')
        try:
            return paginator.paginate(**kwargs).build_full_result()
        except is_boto3_error_code('ClusterNotFoundException') as e:
            self.module.fail_json_aws(e, 'Could not find cluster to list services')

    @AWSRetry.jittered_backoff(retries=5, delay=5, backoff=2.0)
    def describe_services_with_backoff(self, **kwargs):
        return self.ecs.describe_services(**kwargs)

    def list_services(self, cluster):
        fn_args = dict()
        if cluster and cluster is not None:
            fn_args['cluster'] = cluster
        try:
            response = self.list_services_with_backoff(**fn_args)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg="Couldn't list ECS services")
        relevant_response = dict(services=response['serviceArns'])
        return relevant_response

    def describe_services(self, cluster, services):
        fn_args = dict()
        if cluster and cluster is not None:
            fn_args['cluster'] = cluster
        fn_args['services'] = services
        try:
            response = self.describe_services_with_backoff(**fn_args)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg="Couldn't describe ECS services")
        running_services = [self.extract_service_from(service) for service in response.get('services', [])]
        services_not_running = response.get('failures', [])
        return (running_services, services_not_running)

    def extract_service_from(self, service):
        if 'deployments' in service:
            for d in service['deployments']:
                if 'createdAt' in d:
                    d['createdAt'] = str(d['createdAt'])
                if 'updatedAt' in d:
                    d['updatedAt'] = str(d['updatedAt'])
        if 'events' in service:
            if not self.module.params['events']:
                del service['events']
            else:
                for e in service['events']:
                    if 'createdAt' in e:
                        e['createdAt'] = str(e['createdAt'])
        return service