import json
import logging
import os
import platform
import signal
import sys
import tarfile
import time
import urllib.parse
from subprocess import Popen
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.container import (
from mlflow.models.container import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_unique_resource_id
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data
def _update_sagemaker_endpoint(endpoint_name, model_name, model_uri, image_url, model_s3_path, flavor, instance_type, instance_count, vpc_config, mode, role, sage_client, s3_client, variant_name=None, async_inference_config=None, serverless_config=None, data_capture_config=None, env=None, tags=None):
    """
    Args:
        endpoint_name: The name of the SageMaker endpoint to update.
        model_name: The name to assign the new SageMaker model that will be associated with the
            specified endpoint.
        model_uri: URI of the MLflow model to associate with the specified SageMaker endpoint.
        image_url: URL of the ECR-hosted Docker image the model is being deployed into
        model_s3_path: S3 path where we stored the model artifacts
        flavor: The name of the flavor of the model to use for deployment.
        instance_type: The type of SageMaker ML instance on which to deploy the model.
        instance_count: The number of SageMaker ML instances on which to deploy the model.
        vpc_config: A dictionary specifying the VPC configuration to use when creating the
            new SageMaker model associated with this SageMaker endpoint.
        mode: either mlflow.sagemaker.DEPLOYMENT_MODE_ADD or
            mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE.
        role: SageMaker execution ARN role.
        sage_client: A boto3 client for SageMaker.
        s3_client: A boto3 client for S3.
        variant_name: The name to assign to the new production variant if it doesn't already exist.
        async_inference_config: A dictionary specifying the async inference configuration to use.
            For more information, see https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AsyncInferenceConfig.html.
            Defaults to ``None``.
        data_capture_config: A dictionary specifying the data capture configuration to use.
            For more information, see https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DataCaptureConfig.html.
            Defaults to ``None``.
        env: A dictionary of environment variables to set for the model.
        tags: A dictionary of tags to apply to the endpoint configuration.
    """
    if mode not in [DEPLOYMENT_MODE_ADD, DEPLOYMENT_MODE_REPLACE]:
        msg = f'Invalid mode `{mode}` for deployment to a pre-existing application'
        raise ValueError(msg)
    endpoint_info = sage_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_arn = endpoint_info['EndpointArn']
    deployed_config_name = endpoint_info['EndpointConfigName']
    deployed_config_info = sage_client.describe_endpoint_config(EndpointConfigName=deployed_config_name)
    deployed_config_arn = deployed_config_info['EndpointConfigArn']
    deployed_production_variants = deployed_config_info['ProductionVariants']
    _logger.info('Found active endpoint with arn: %s. Updating...', endpoint_arn)
    new_model_response = _create_sagemaker_model(model_name=model_name, model_s3_path=model_s3_path, model_uri=model_uri, flavor=flavor, vpc_config=vpc_config, image_url=image_url, execution_role=role, sage_client=sage_client, env=env or {}, tags=tags or {})
    _logger.info('Created new model with arn: %s', new_model_response['ModelArn'])
    if not variant_name:
        variant_name = model_name
    if mode == DEPLOYMENT_MODE_ADD:
        new_model_weight = 0
        production_variants = deployed_production_variants
    elif mode == DEPLOYMENT_MODE_REPLACE:
        new_model_weight = 1
        production_variants = []
    new_production_variant = {'VariantName': variant_name, 'ModelName': model_name, 'InitialVariantWeight': new_model_weight}
    if serverless_config:
        new_production_variant['ServerlessConfig'] = serverless_config
    else:
        new_production_variant['InstanceType'] = instance_type
        new_production_variant['InitialInstanceCount'] = instance_count
    production_variants.append(new_production_variant)
    new_config_name = _get_sagemaker_config_name(endpoint_name)
    config_tags = _get_sagemaker_config_tags(endpoint_name)
    endpoint_config_kwargs = {'EndpointConfigName': new_config_name, 'ProductionVariants': production_variants, 'Tags': config_tags}
    if async_inference_config:
        endpoint_config_kwargs['AsyncInferenceConfig'] = async_inference_config
    if data_capture_config is not None:
        endpoint_config_kwargs['DataCaptureConfig'] = data_capture_config
    endpoint_config_response = sage_client.create_endpoint_config(**endpoint_config_kwargs)
    _logger.info('Created new endpoint configuration with arn: %s', endpoint_config_response['EndpointConfigArn'])
    sage_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=new_config_name)
    _logger.info('Updated endpoint with new configuration!')
    operation_start_time = time.time()

    def status_check_fn():
        if time.time() - operation_start_time < 20:
            return _SageMakerOperationStatus.in_progress()
        endpoint_info = sage_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_update_was_rolled_back = endpoint_info['EndpointStatus'] == 'InService' and endpoint_info['EndpointConfigName'] != new_config_name
        if endpoint_update_was_rolled_back or endpoint_info['EndpointStatus'] == 'Failed':
            failure_reason = endpoint_info.get('FailureReason', 'An unknown SageMaker failure occurred. Please see the SageMaker console logs for more information.')
            return _SageMakerOperationStatus.failed(failure_reason)
        elif endpoint_info['EndpointStatus'] == 'InService':
            return _SageMakerOperationStatus.succeeded('The SageMaker endpoint was updated successfully.')
        else:
            return _SageMakerOperationStatus.in_progress('The update operation is still in progress. Current endpoint status: "{endpoint_status}"'.format(endpoint_status=endpoint_info['EndpointStatus']))

    def cleanup_fn():
        _logger.info('Cleaning up unused resources...')
        if mode == DEPLOYMENT_MODE_REPLACE:
            for pv in deployed_production_variants:
                deployed_model_arn = _delete_sagemaker_model(model_name=pv['ModelName'], sage_client=sage_client, s3_client=s3_client)
                _logger.info('Deleted model with arn: %s', deployed_model_arn)
        sage_client.delete_endpoint_config(EndpointConfigName=deployed_config_name)
        _logger.info('Deleted endpoint configuration with arn: %s', deployed_config_arn)
    return _SageMakerOperation(status_check_fn=status_check_fn, cleanup_fn=cleanup_fn)