import json
import os
from ...utils.constants import SAGEMAKER_PARALLEL_EC2_INSTANCES, TORCH_DYNAMO_MODES
from ...utils.dataclasses import ComputeEnvironment, SageMakerDistributedType
from ...utils.imports import is_boto3_available
from .config_args import SageMakerConfig
from .config_utils import (
def _create_iam_role_for_sagemaker(role_name):
    iam_client = boto3.client('iam')
    sagemaker_trust_policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'sagemaker.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
    try:
        iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(sagemaker_trust_policy, indent=2))
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['sagemaker:*', 'ecr:GetDownloadUrlForLayer', 'ecr:BatchGetImage', 'ecr:BatchCheckLayerAvailability', 'ecr:GetAuthorizationToken', 'cloudwatch:PutMetricData', 'cloudwatch:GetMetricData', 'cloudwatch:GetMetricStatistics', 'cloudwatch:ListMetrics', 'logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:DescribeLogStreams', 'logs:PutLogEvents', 'logs:GetLogEvents', 's3:CreateBucket', 's3:ListBucket', 's3:GetBucketLocation', 's3:GetObject', 's3:PutObject'], 'Resource': '*'}]}
        iam_client.put_role_policy(RoleName=role_name, PolicyName=f'{role_name}_policy_permission', PolicyDocument=json.dumps(policy_document, indent=2))
    except iam_client.exceptions.EntityAlreadyExistsException:
        print(f'role {role_name} already exists. Using existing one')