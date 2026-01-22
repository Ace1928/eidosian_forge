from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule

    Args:
        conn (boto3.AutoScaling.Client): Valid Boto3 ASG client.
        module: AnsibleAWSModule object

    Returns:
        {
            "instance_refreshes": [
                    {
                        'auto_scaling_group_name': 'ansible-test-hermes-63642726-asg',
                        'instance_refresh_id': '6507a3e5-4950-4503-8978-e9f2636efc09',
                        'instances_to_update': 1,
                        'percentage_complete': 0,
                        "preferences": {
                            "instance_warmup": 60,
                            "min_healthy_percentage": 90,
                            "skip_matching": false
                        },
                        'start_time': '2021-02-04T03:39:40+00:00',
                        'status': 'Cancelling',
                        'status_reason': 'Replacing instances before cancelling.',
                    }
              ]
        }
    