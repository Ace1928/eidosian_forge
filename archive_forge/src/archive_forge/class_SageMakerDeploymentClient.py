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
class SageMakerDeploymentClient(BaseDeploymentClient):
    """
    Initialize a deployment client for SageMaker. The default region and assumed role ARN will
    be set according to the value of the `target_uri`.

    This class is meant to supercede the other ``mlflow.sagemaker`` real-time serving API's.
    It is also designed to be used through the :py:mod:`mlflow.deployments` module.
    This means that you can deploy to SageMaker using the
    `mlflow deployments CLI <https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments>`_ and
    get a client through the :py:mod:`mlflow.deployments.get_deploy_client` function.

    Args:
        target_uri: A URI that follows one of the following formats:

            - ``sagemaker``: This will set the default region to `us-west-2` and
              the default assumed role ARN to `None`.
            - ``sagemaker:/region_name``: This will set the default region to
              `region_name` and the default assumed role ARN to `None`.
            - ``sagemaker:/region_name/assumed_role_arn``: This will set the default
              region to `region_name` and the default assumed role ARN to
              `assumed_role_arn`.

            When an `assumed_role_arn` is provided without a `region_name`,
            an MlflowException will be raised.
    """

    def __init__(self, target_uri):
        super().__init__(target_uri=target_uri)
        self.region_name = DEFAULT_REGION_NAME
        self.assumed_role_arn = None
        self._get_values_from_target_uri()

    def _get_values_from_target_uri(self):
        parsed = urllib.parse.urlparse(self.target_uri)
        values_str = parsed.path.strip('/')
        if not parsed.scheme or not values_str:
            return
        separator_index = values_str.find('/')
        if separator_index == -1:
            self.region_name = values_str
        else:
            self.region_name = values_str[:separator_index]
            self.assumed_role_arn = values_str[separator_index + 1:]
            self.assumed_role_arn = self.assumed_role_arn.strip('/')
        if self.region_name.startswith('arn'):
            raise MlflowException(message=f'It looks like the target_uri contains an IAM role ARN without a region name.\nA region name must be provided when the target_uri contains a role ARN.\nIn this case, the target_uri must follow the format: sagemaker:/region_name/assumed_role_arn.\nThe provided target_uri is: {self.target_uri}\n', error_code=INVALID_PARAMETER_VALUE)

    def _default_deployment_config(self, create_mode=True):
        config = {'assume_role_arn': self.assumed_role_arn, 'execution_role_arn': None, 'bucket': None, 'image_url': None, 'region_name': self.region_name, 'archive': False, 'instance_type': DEFAULT_SAGEMAKER_INSTANCE_TYPE, 'instance_count': DEFAULT_SAGEMAKER_INSTANCE_COUNT, 'vpc_config': None, 'data_capture_config': None, 'synchronous': True, 'timeout_seconds': 1200, 'variant_name': None, 'env': None, 'tags': None, 'async_inference_config': {}, 'serverless_config': {}}
        if create_mode:
            config['mode'] = DEPLOYMENT_MODE_CREATE
        else:
            config['mode'] = DEPLOYMENT_MODE_REPLACE
        return config

    def _apply_custom_config(self, config, custom_config):
        int_fields = {'instance_count', 'timeout_seconds'}
        bool_fields = {'synchronous', 'archive'}
        dict_fields = {'vpc_config', 'data_capture_config', 'tags', 'env', 'async_inference_config', 'serverless_config'}
        for key, value in custom_config.items():
            if key not in config:
                continue
            if key in int_fields and (not isinstance(value, int)):
                value = int(value)
            elif key in bool_fields and (not isinstance(value, bool)):
                value = value == 'True'
            elif key in dict_fields and (not isinstance(value, dict)):
                value = json.loads(value)
            config[key] = value

    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        Deploy an MLflow model on AWS SageMaker.
        The currently active AWS account must have correct permissions set up.

        This function creates a SageMaker endpoint. For more information about the input data
        formats accepted by this endpoint, see the
        :ref:`MLflow deployment tools documentation <sagemaker_deployment>`.

        Args:
            name: Name of the deployed application.
            model_uri: The location, in URI format, of the MLflow model to deploy to SageMaker.
                For example:

                - ``/Users/me/path/to/local/model``
                - ``relative/path/to/local/model``
                - ``s3://my_bucket/path/to/model``
                - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                - ``models:/<model_name>/<model_version>``
                - ``models:/<model_name>/<stage>``

                For more information about supported URI schemes, see
                `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                artifact-locations>`_.
            flavor: The name of the flavor of the model to use for deployment. Must be either
                ``None`` or one of mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS.
                If ``None``, a flavor is automatically selected from the model's available
                flavors. If the specified flavor is not present or not supported for
                deployment, an exception will be thrown.
            config: Configuration parameters. The supported parameters are:

                - ``assume_role_arn``: The name of an IAM cross-account role to be assumed
                  to deploy SageMaker to another AWS account. If this parameter is not
                  specified, the role given in the ``target_uri`` will be used. If the
                  role is not given in the ``target_uri``, defaults to ``us-west-2``.

                - ``execution_role_arn``: The name of an IAM role granting the SageMaker
                  service permissions to access the specified Docker image and S3 bucket
                  containing MLflow model artifacts. If unspecified, the currently-assumed
                  role will be used. This execution role is passed to the SageMaker service
                  when creating a SageMaker model from the specified MLflow model. It is
                  passed as the ``ExecutionRoleArn`` parameter of the `SageMaker
                  CreateModel API call <https://docs.aws.amazon.com/sagemaker/latest/
                  dg/API_CreateModel.html>`_. This role is *not* assumed for any other
                  call. For more information about SageMaker execution roles for model
                  creation, see
                  https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html.

                - ``bucket``: S3 bucket where model artifacts will be stored. Defaults to a
                  SageMaker-compatible bucket name.

                - ``image_url``: URL of the ECR-hosted Docker image the model should be
                  deployed into, produced by ``mlflow sagemaker build-and-push-container``.
                  This parameter can also be specified by the environment variable
                  ``MLFLOW_SAGEMAKER_DEPLOY_IMG_URL``.

                - ``region_name``: Name of the AWS region to which to deploy the application.
                  If unspecified, use the region name given in the ``target_uri``.
                  If it is also not specified in the ``target_uri``,
                  defaults to ``us-west-2``.

                - ``archive``: If ``True``, any pre-existing SageMaker application resources
                  that become inactive (i.e. as a result of deploying in
                  ``mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE`` mode) are preserved.
                  These resources may include unused SageMaker models and endpoint
                  configurations that were associated with a prior version of the
                  application endpoint. If ``False``, these resources are deleted.
                  In order to use ``archive=False``, ``create_deployment()`` must be executed
                  synchronously with ``synchronous=True``. Defaults to ``False``.

                - ``instance_type``: The type of SageMaker ML instance on which to deploy the
                  model. For a list of supported instance types, see
                  https://aws.amazon.com/sagemaker/pricing/instance-types/.
                  Defaults to ``ml.m4.xlarge``.

                - ``instance_count``: The number of SageMaker ML instances on which to deploy
                  the model. Defaults to ``1``.

                - ``synchronous``: If ``True``, this function will block until the deployment
                  process succeeds or encounters an irrecoverable failure. If ``False``,
                  this function will return immediately after starting the deployment
                  process. It will not wait for the deployment process to complete;
                  in this case, the caller is responsible for monitoring the health and
                  status of the pending deployment via native SageMaker APIs or the AWS
                  console. Defaults to ``True``.

                - ``timeout_seconds``: If ``synchronous`` is ``True``, the deployment process
                  will return after the specified number of seconds if no definitive result
                  (success or failure) is achieved. Once the function returns, the caller is
                  responsible for monitoring the health and status of the pending
                  deployment using native SageMaker APIs or the AWS console. If
                  ``synchronous`` is ``False``, this parameter is ignored.
                  Defaults to ``300``.

                - ``vpc_config``: A dictionary specifying the VPC configuration to use when
                  creating the new SageMaker model associated with this application.
                  The acceptable values for this parameter are identical to those of the
                  ``VpcConfig`` parameter in the `SageMaker boto3 client's create_model
                  method <https://boto3.readthedocs.io/en/latest/reference/services/sagemaker.html
                  #SageMaker.Client.create_model>`_. For more information, see
                  https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html.
                  Defaults to ``None``.

                - ``data_capture_config``: A dictionary specifying the data capture
                  configuration to use when creating the new SageMaker model associated with
                  this application.
                  For more information, see
                  https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DataCaptureConfig.html.
                  Defaults to ``None``.

                - ``variant_name``: A string specifying the desired name when creating a production
                  variant.  Defaults to ``None``.

                - ``async_inference_config``: A dictionary specifying the
                  async_inference_configuration

                - ``serverless_config``: A dictionary specifying the serverless_configuration

                - ``env``: A dictionary specifying environment variables as key-value
                  pairs to be set for the deployed model. Defaults to ``None``.

                - ``tags``: A dictionary of key-value pairs representing additional
                  tags to be set for the deployed model. Defaults to ``None``.

            endpoint: (optional) Endpoint to create the deployment under. Currently unsupported

        .. code-block:: python
            :caption: Python example

            from mlflow.deployments import get_deploy_client

            vpc_config = {
                "SecurityGroupIds": [
                    "sg-123456abc",
                ],
                "Subnets": [
                    "subnet-123456abc",
                ],
            }
            config = dict(
                assume_role_arn="arn:aws:123:role/assumed_role",
                execution_role_arn="arn:aws:456:role/execution_role",
                bucket_name="my-s3-bucket",
                image_url="1234.dkr.ecr.us-east-1.amazonaws.com/mlflow-test:1.23.1",
                region_name="us-east-1",
                archive=False,
                instance_type="ml.m5.4xlarge",
                instance_count=1,
                synchronous=True,
                timeout_seconds=300,
                vpc_config=vpc_config,
                variant_name="prod-variant-1",
                env={"DISABLE_NGINX": "true", "GUNICORN_CMD_ARGS": '"--timeout 60"'},
                tags={"training_timestamp": "2022-11-01T05:12:26"},
            )
            client = get_deploy_client("sagemaker")
            client.create_deployment(
                "my-deployment",
                model_uri="/mlruns/0/abc/model",
                flavor="python_function",
                config=config,
            )
        .. code-block:: bash
            :caption:  Command-line example

            mlflow deployments create --target sagemaker:/us-east-1/arn:aws:123:role/assumed_role \\
                    --name my-deployment \\
                    --model-uri /mlruns/0/abc/model \\
                    --flavor python_function\\
                    -C execution_role_arn=arn:aws:456:role/execution_role \\
                    -C bucket_name=my-s3-bucket \\
                    -C image_url=1234.dkr.ecr.us-east-1.amazonaws.com/mlflow-test:1.23.1 \\
                    -C region_name=us-east-1 \\
                    -C archive=False \\
                    -C instance_type=ml.m5.4xlarge \\
                    -C instance_count=1 \\
                    -C synchronous=True \\
                    -C timeout_seconds=300 \\
                    -C variant_name=prod-variant-1 \\
                    -C vpc_config='{"SecurityGroupIds": ["sg-123456abc"], \\
                    "Subnets": ["subnet-123456abc"]}' \\
                    -C data_capture_config='{"EnableCapture": True, \\
                    'InitalSamplingPercentage': 100, 'DestinationS3Uri": 's3://my-bucket/path', \\
                    'CaptureOptions': [{'CaptureMode': 'Output'}]}'
                    -C env='{"DISABLE_NGINX": "true", "GUNICORN_CMD_ARGS": ""--timeout 60""}' \\
                    -C tags='{"training_timestamp": "2022-11-01T05:12:26"}' \\
        """
        final_config = self._default_deployment_config()
        if config:
            self._apply_custom_config(final_config, config)
        app_name, flavor = _deploy(app_name=name, model_uri=model_uri, flavor=flavor, execution_role_arn=final_config['execution_role_arn'], assume_role_arn=final_config['assume_role_arn'], bucket=final_config['bucket'], image_url=final_config['image_url'], region_name=final_config['region_name'], mode=mlflow.sagemaker.DEPLOYMENT_MODE_CREATE, archive=final_config['archive'], instance_type=final_config['instance_type'], instance_count=final_config['instance_count'], vpc_config=final_config['vpc_config'], data_capture_config=final_config['data_capture_config'], synchronous=final_config['synchronous'], timeout_seconds=final_config['timeout_seconds'], variant_name=final_config['variant_name'], async_inference_config=final_config['async_inference_config'], serverless_config=final_config['serverless_config'], env=final_config['env'], tags=final_config['tags'])
        return {'name': app_name, 'flavor': flavor}

    def update_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        Update a deployment on AWS SageMaker. This function can replace or add a new model to
        an existing SageMaker endpoint. By default, this function replaces the existing model
        with the new one. The currently active AWS account must have correct permissions set up.

        Args:
            name: Name of the deployed application.
            model_uri: The location, in URI format, of the MLflow model to deploy to SageMaker.
                For example:

                - ``/Users/me/path/to/local/model``
                - ``relative/path/to/local/model``
                - ``s3://my_bucket/path/to/model``
                - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                - ``models:/<model_name>/<model_version>``
                - ``models:/<model_name>/<stage>``

                For more information about supported URI schemes, see
                `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                artifact-locations>`_.

            flavor: The name of the flavor of the model to use for deployment. Must be either
                ``None`` or one of mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS.
                If ``None``, a flavor is automatically selected from the model's available
                flavors. If the specified flavor is not present or not supported for
                deployment, an exception will be thrown.

            config: Configuration parameters. The supported parameters are:

                - ``assume_role_arn``: The name of an IAM cross-account role to be assumed
                  to deploy SageMaker to another AWS account. If this parameter is not
                  specified, the role given in the ``target_uri`` will be used. If the
                  role is not given in the ``target_uri``, defaults to ``us-west-2``.

                - ``execution_role_arn``: The name of an IAM role granting the SageMaker
                  service permissions to access the specified Docker image and S3 bucket
                  containing MLflow model artifacts. If unspecified, the currently-assumed
                  role will be used. This execution role is passed to the SageMaker service
                  when creating a SageMaker model from the specified MLflow model. It is
                  passed as the ``ExecutionRoleArn`` parameter of the `SageMaker
                  CreateModel API call <https://docs.aws.amazon.com/sagemaker/latest/
                  dg/API_CreateModel.html>`_. This role is *not* assumed for any other
                  call. For more information about SageMaker execution roles for model
                  creation, see
                  https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html.

                - ``bucket``: S3 bucket where model artifacts will be stored. Defaults to a
                  SageMaker-compatible bucket name.

                - ``image_url``: URL of the ECR-hosted Docker image the model should be
                  deployed into, produced by ``mlflow sagemaker build-and-push-container``.
                  This parameter can also be specified by the environment variable
                  ``MLFLOW_SAGEMAKER_DEPLOY_IMG_URL``.

                - ``region_name``: Name of the AWS region to which to deploy the application.
                  If unspecified, use the region name given in the ``target_uri``.
                  If it is also not specified in the ``target_uri``,
                  defaults to ``us-west-2``.

                - ``mode``: The mode in which to deploy the application.
                  Must be one of the following:

                  ``mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE``
                      If an application of the specified name exists, its model(s) is
                      replaced with the specified model. If no such application exists,
                      it is created with the specified name and model.
                      This is the default mode.

                  ``mlflow.sagemaker.DEPLOYMENT_MODE_ADD``
                      Add the specified model to a pre-existing application with the
                      specified name, if one exists. If the application does not exist,
                      a new application is created with the specified name and model.
                      NOTE: If the application **already exists**, the specified model is
                      added to the application's corresponding SageMaker endpoint with an
                      initial weight of zero (0). To route traffic to the model,
                      update the application's associated endpoint configuration using
                      either the AWS console or the ``UpdateEndpointWeightsAndCapacities``
                      function defined in https://docs.aws.amazon.com/sagemaker/latest/dg/API_UpdateEndpointWeightsAndCapacities.html.

                - ``archive``: If ``True``, any pre-existing SageMaker application resources
                  that become inactive (i.e. as a result of deploying in
                  ``mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE`` mode) are preserved.
                  These resources may include unused SageMaker models and endpoint
                  configurations that were associated with a prior version of the
                  application endpoint. If ``False``, these resources are deleted.
                  In order to use ``archive=False``, ``update_deployment()`` must be executed
                  synchronously with ``synchronous=True``. Defaults to ``False``.

                - ``instance_type``: The type of SageMaker ML instance on which to deploy the
                  model. For a list of supported instance types, see
                  https://aws.amazon.com/sagemaker/pricing/instance-types/.
                  Defaults to ``ml.m4.xlarge``.

                - ``instance_count``: The number of SageMaker ML instances on which to deploy
                  the model. Defaults to ``1``.

                - ``synchronous``: If ``True``, this function will block until the deployment
                  process succeeds or encounters an irrecoverable failure. If ``False``,
                  this function will return immediately after starting the deployment
                  process. It will not wait for the deployment process to complete;
                  in this case, the caller is responsible for monitoring the health and
                  status of the pending deployment via native SageMaker APIs or the AWS
                  console. Defaults to ``True``.

                - ``timeout_seconds``: If ``synchronous`` is ``True``, the deployment process
                  will return after the specified number of seconds if no definitive result
                  (success or failure) is achieved. Once the function returns, the caller is
                  responsible for monitoring the health and status of the pending
                  deployment using native SageMaker APIs or the AWS console. If
                  ``synchronous`` is ``False``, this parameter is ignored.
                  Defaults to ``300``.

                - ``variant_name``: A string specifying the desired name when creating a
                  production variant.  Defaults to ``None``.

                - ``vpc_config``: A dictionary specifying the VPC configuration to use when
                  creating the new SageMaker model associated with this application.
                  The acceptable values for this parameter are identical to those of the
                  ``VpcConfig`` parameter in the `SageMaker boto3 client's create_model
                  method <https://boto3.readthedocs.io/en/latest/reference/services/sagemaker.html
                  #SageMaker.Client.create_model>`_. For more information, see
                  https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html.
                  Defaults to ``None``.

                - ``data_capture_config``: A dictionary specifying the data capture
                  configuration to use when creating the new SageMaker model associated with
                  this application.
                  For more information, see
                  https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DataCaptureConfig.html.
                  Defaults to ``None``.

                - ``async_inference_config``: A dictionary specifying the async config
                  configuration. Defaults to ``None``.

                - ``env``: A dictionary specifying environment variables as key-value pairs
                  to be set for the deployed model. Defaults to ``None``.

                - ``tags``: A dictionary of key-value pairs representing additional tags
                  to be set for the deployed model. Defaults to ``None``.

            endpoint: (optional) Endpoint containing the deployment to update. Currently unsupported

        .. code-block:: python
            :caption: Python example

            from mlflow.deployments import get_deploy_client

            vpc_config = {
                "SecurityGroupIds": [
                    "sg-123456abc",
                ],
                "Subnets": [
                    "subnet-123456abc",
                ],
            }
            data_capture_config = {
                "EnableCapture": True,
                "InitalSamplingPercentage": 100,
                "DestinationS3Uri": "s3://my-bucket/path",
                "CaptureOptions": [{"CaptureMode": "Output"}],
            }
            config = dict(
                assume_role_arn="arn:aws:123:role/assumed_role",
                execution_role_arn="arn:aws:456:role/execution_role",
                bucket_name="my-s3-bucket",
                image_url="1234.dkr.ecr.us-east-1.amazonaws.com/mlflow-test:1.23.1",
                region_name="us-east-1",
                mode="replace",
                archive=False,
                instance_type="ml.m5.4xlarge",
                instance_count=1,
                synchronous=True,
                timeout_seconds=300,
                variant_name="prod-variant-1",
                vpc_config=vpc_config,
                data_capture_config=data_capture_config,
                env={"DISABLE_NGINX": "true", "GUNICORN_CMD_ARGS": '"--timeout 60"'},
                tags={"training_timestamp": "2022-11-01T05:12:26"},
            )
            client = get_deploy_client("sagemaker")
            client.update_deployment(
                "my-deployment",
                model_uri="/mlruns/0/abc/model",
                flavor="python_function",
                config=config,
            )
        .. code-block:: bash
            :caption:  Command-line example

            mlflow deployments update --target sagemaker:/us-east-1/arn:aws:123:role/assumed_role \\
                    --name my-deployment \\
                    --model-uri /mlruns/0/abc/model \\
                    --flavor python_function\\
                    -C execution_role_arn=arn:aws:456:role/execution_role \\
                    -C bucket_name=my-s3-bucket \\
                    -C image_url=1234.dkr.ecr.us-east-1.amazonaws.com/mlflow-test:1.23.1 \\
                    -C region_name=us-east-1 \\
                    -C mode=replace \\
                    -C archive=False \\
                    -C instance_type=ml.m5.4xlarge \\
                    -C instance_count=1 \\
                    -C synchronous=True \\
                    -C timeout_seconds=300 \\
                    -C variant_name=prod-variant-1 \\
                    -C vpc_config='{"SecurityGroupIds": ["sg-123456abc"], \\
                    "Subnets": ["subnet-123456abc"]}' \\
                    -C data_capture_config='{"EnableCapture": True, \\
                    "InitalSamplingPercentage": 100, "DestinationS3Uri": "s3://my-bucket/path", \\
                    "CaptureOptions": [{"CaptureMode": "Output"}]}'
                    -C env='{"DISABLE_NGINX": "true", "GUNICORN_CMD_ARGS": ""--timeout 60""}' \\
                    -C tags='{"training_timestamp": "2022-11-01T05:12:26"}' \\
        """
        final_config = self._default_deployment_config(create_mode=False)
        if config:
            self._apply_custom_config(final_config, config)
        if model_uri is None:
            raise MlflowException(message='A model_uri must be provided when updating a SageMaker deployment', error_code=INVALID_PARAMETER_VALUE)
        if final_config['mode'] not in [DEPLOYMENT_MODE_ADD, DEPLOYMENT_MODE_REPLACE]:
            raise MlflowException(message=f'Invalid mode `{final_config['mode']}` for deployment to a pre-existing application', error_code=INVALID_PARAMETER_VALUE)
        app_name, flavor = _deploy(app_name=name, model_uri=model_uri, flavor=flavor, execution_role_arn=final_config['execution_role_arn'], assume_role_arn=final_config['assume_role_arn'], bucket=final_config['bucket'], image_url=final_config['image_url'], region_name=final_config['region_name'], mode=final_config['mode'], archive=final_config['archive'], instance_type=final_config['instance_type'], instance_count=final_config['instance_count'], vpc_config=final_config['vpc_config'], data_capture_config=final_config['data_capture_config'], synchronous=final_config['synchronous'], timeout_seconds=final_config['timeout_seconds'], variant_name=final_config['variant_name'], async_inference_config=final_config['async_inference_config'], serverless_config=final_config['serverless_config'], env=final_config['env'], tags=final_config['tags'])
        return {'name': app_name, 'flavor': flavor}

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        Delete a SageMaker application.

        Args:
            name: Name of the deployed application.
            config: Configuration parameters. The supported parameters are:

                - ``assume_role_arn``: The name of an IAM role to be assumed to delete
                  the SageMaker deployment.

                - ``region_name``: Name of the AWS region in which the application
                  is deployed. Defaults to ``us-west-2`` or the region provided in
                  the `target_uri`.

                - ``archive``: If `True`, resources associated with the specified
                  application, such as its associated models and endpoint configuration,
                  are preserved. If `False`, these resources are deleted. In order to use
                  ``archive=False``, ``delete()`` must be executed synchronously with
                  ``synchronous=True``. Defaults to ``False``.

                - ``synchronous``: If `True`, this function blocks until the deletion process
                  succeeds or encounters an irrecoverable failure. If `False`, this function
                  returns immediately after starting the deletion process. It will not wait
                  for the deletion process to complete; in this case, the caller is
                  responsible for monitoring the status of the deletion process via native
                  SageMaker APIs or the AWS console. Defaults to ``True``.

                - ``timeout_seconds``: If `synchronous` is `True`, the deletion process
                  returns after the specified number of seconds if no definitive result
                  (success or failure) is achieved. Once the function returns, the caller
                  is responsible for monitoring the status of the deletion process via native
                  SageMaker APIs or the AWS console. If `synchronous` is False, this
                  parameter is ignored. Defaults to ``300``.

            endpoint: (optional) Endpoint containing the deployment to delete. Currently unsupported

        .. code-block:: python
            :caption: Python example

            from mlflow.deployments import get_deploy_client

            config = dict(
                assume_role_arn="arn:aws:123:role/assumed_role",
                region_name="us-east-1",
                archive=False,
                synchronous=True,
                timeout_seconds=300,
            )
            client = get_deploy_client("sagemaker")
            client.delete_deployment("my-deployment", config=config)

        .. code-block:: bash
            :caption: Command-line example

            mlflow deployments delete --target sagemaker \\
                    --name my-deployment \\
                    -C assume_role_arn=arn:aws:123:role/assumed_role \\
                    -C region_name=us-east-1 \\
                    -C archive=False \\
                    -C synchronous=True \\
                    -C timeout_seconds=300
        """
        final_config = {'region_name': self.region_name, 'archive': False, 'synchronous': True, 'timeout_seconds': 300, 'assume_role_arn': self.assumed_role_arn}
        if config:
            self._apply_custom_config(final_config, config)
        _delete(name, region_name=final_config['region_name'], assume_role_arn=final_config['assume_role_arn'], archive=final_config['archive'], synchronous=final_config['synchronous'], timeout_seconds=final_config['timeout_seconds'])

    def list_deployments(self, endpoint=None):
        """
        List deployments. This method returns a list of dictionaries that describes each deployment.

        If a region name needs to be specified, the plugin must be initialized
        with the AWS region in the ``target_uri`` such as ``sagemaker:/us-east-1``.

        To assume an IAM role, the plugin must be initialized
        with the AWS region and the role ARN in the ``target_uri`` such as
        ``sagemaker:/us-east-1/arn:aws:1234:role/assumed_role``.

        Args:
            endpoint: (optional) List deployments in the specified endpoint. Currently unsupported

        Returns:
            A list of dictionaries corresponding to deployments.

        .. code-block:: python
            :caption: Python example

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("sagemaker:/us-east-1/arn:aws:123:role/assumed_role")
            client.list_deployments()

        .. code-block:: bash
            :caption: Command-line example

            mlflow deployments list --target sagemaker:/us-east-1/arn:aws:1234:role/assumed_role
        """
        import boto3
        assume_role_credentials = _assume_role_and_get_credentials(assume_role_arn=self.assumed_role_arn)
        sage_client = boto3.client('sagemaker', region_name=self.region_name, **assume_role_credentials)
        return sage_client.list_endpoints()['Endpoints']

    def get_deployment(self, name, endpoint=None):
        """
        Returns a dictionary describing the specified deployment.

        If a region name needs to be specified, the plugin must be initialized
        with the AWS region in the ``target_uri`` such as ``sagemaker:/us-east-1``.

        To assume an IAM role, the plugin must be initialized
        with the AWS region and the role ARN in the ``target_uri`` such as
        ``sagemaker:/us-east-1/arn:aws:1234:role/assumed_role``.

        A :py:class:`mlflow.exceptions.MlflowException` will also be thrown when an error occurs
        while retrieving the deployment.

        Args:
            name: Name of deployment to retrieve
            endpoint: (optional) Endpoint containing the deployment to get. Currently unsupported

        Returns:
            A dictionary that describes the specified deployment

        .. code-block:: python
            :caption: Python example

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("sagemaker:/us-east-1/arn:aws:123:role/assumed_role")
            client.get_deployment("my-deployment")

        .. code-block:: bash
            :caption: Command-line example

            mlflow deployments get --target sagemaker:/us-east-1/arn:aws:1234:role/assumed_role \\
                --name my-deployment
        """
        import boto3
        assume_role_credentials = _assume_role_and_get_credentials(assume_role_arn=self.assumed_role_arn)
        try:
            sage_client = boto3.client('sagemaker', region_name=self.region_name, **assume_role_credentials)
            return sage_client.describe_endpoint(EndpointName=name)
        except Exception as exc:
            raise MlflowException(message=f'There was an error while retrieving the deployment: {exc}\n')

    def predict(self, deployment_name=None, inputs=None, endpoint=None, params: Optional[Dict[str, Any]]=None):
        """
        Compute predictions from the specified deployment using the provided PyFunc input.

        The input/output types of this method match the :ref:`MLflow PyFunc prediction
        interface <pyfunc-inference-api>`.

        If a region name needs to be specified, the plugin must be initialized
        with the AWS region in the ``target_uri`` such as ``sagemaker:/us-east-1``.

        To assume an IAM role, the plugin must be initialized
        with the AWS region and the role ARN in the ``target_uri`` such as
        ``sagemaker:/us-east-1/arn:aws:1234:role/assumed_role``.

        Args:
            deployment_name: Name of the deployment to predict against.
            inputs: Input data (or arguments) to pass to the deployment or model endpoint for
                inference. For a complete list of supported input types, see
                :ref:`pyfunc-inference-api`.
            endpoint: Endpoint to predict against. Currently unsupported

        Returns:
            A PyFunc output, such as a Pandas DataFrame, Pandas Series, or NumPy array.
            For a complete list of supported output types, see :ref:`pyfunc-inference-api`.

        .. code-block:: python
            :caption: Python example

            import pandas as pd
            from mlflow.deployments import get_deploy_client

            df = pd.DataFrame(data=[[1, 2, 3]], columns=["feat1", "feat2", "feat3"])
            client = get_deploy_client("sagemaker:/us-east-1/arn:aws:123:role/assumed_role")
            client.predict("my-deployment", df)

        .. code-block:: bash
            :caption: Command-line example

            cat > ./input.json <<- input
            {"feat1": {"0": 1}, "feat2": {"0": 2}, "feat3": {"0": 3}}
            input

            mlflow deployments predict \\
                --target sagemaker:/us-east-1/arn:aws:1234:role/assumed_role \\
                --name my-deployment \\
                --input-path ./input.json
        """
        import boto3
        assume_role_credentials = _assume_role_and_get_credentials(assume_role_arn=self.assumed_role_arn)
        try:
            sage_client = boto3.client('sagemaker-runtime', region_name=self.region_name, **assume_role_credentials)
            response = sage_client.invoke_endpoint(EndpointName=deployment_name, Body=dump_input_data(inputs, inputs_key='instances', params=params), ContentType='application/json')
            response_body = response['Body'].read().decode('utf-8')
            return PredictionsResponse.from_json(response_body)
        except Exception as exc:
            raise MlflowException(message=f'There was an error while getting model prediction: {exc}\n')

    def explain(self, deployment_name=None, df=None, endpoint=None):
        """
        *This function has not been implemented and will be coming in the future.*
        """
        raise NotImplementedError('This function is not implemented yet.')

    def create_endpoint(self, name, config=None):
        """
        Create an endpoint with the specified target. By default, this method should block until
        creation completes (i.e. until it's possible to create a deployment within the endpoint).
        In the case of conflicts (e.g. if it's not possible to create the specified endpoint
        due to conflict with an existing endpoint), raises a
        :py:class:`mlflow.exceptions.MlflowException`. See target-specific plugin documentation
        for additional detail on support for asynchronous creation and other configuration.

        Args:
            name: Unique name to use for endpoint. If another endpoint exists with the same
                        name, raises a :py:class:`mlflow.exceptions.MlflowException`.
            config: (optional) Dict containing target-specific configuration for the endpoint.

        Returns:
            Dict corresponding to created endpoint, which must contain the 'name' key.
        """
        raise NotImplementedError('This function is not implemented yet.')

    def update_endpoint(self, endpoint, config=None):
        """
        Update the endpoint with the specified name. You can update any target-specific attributes
        of the endpoint (via `config`). By default, this method should block until the update
        completes (i.e. until it's possible to create a deployment within the endpoint). See
        target-specific plugin documentation for additional detail on support for asynchronous
        update and other configuration.

        Args:
            endpoint: Unique name of endpoint to update
            config: (optional) dict containing target-specific configuration for the endpoint
        """
        raise NotImplementedError('This function is not implemented yet.')

    def delete_endpoint(self, endpoint):
        """
        Delete the endpoint from the specified target. Deletion should be idempotent (i.e. deletion
        should not fail if retried on a non-existent deployment).

        Args:
            endpoint: Name of endpoint to delete
        """
        raise NotImplementedError('This function is not implemented yet.')

    def list_endpoints(self):
        """
        List endpoints in the specified target. This method is expected to return an
        unpaginated list of all endpoints (an alternative would be to return a dict with
        an 'endpoints' field containing the actual endpoints, with plugins able to specify
        other fields, e.g. a next_page_token field, in the returned dictionary for pagination,
        and to accept a `pagination_args` argument to this method for passing
        pagination-related args).

        Returns:
            A list of dicts corresponding to endpoints. Each dict is guaranteed to
            contain a 'name' key containing the endpoint name. The other fields of
            the returned dictionary and their types may vary across targets.
        """
        raise NotImplementedError('This function is not implemented yet.')

    def get_endpoint(self, endpoint):
        """
        Returns a dictionary describing the specified endpoint, throwing a
        py:class:`mlflow.exception.MlflowException` if no endpoint exists with the provided
        name.
        The dict is guaranteed to contain an 'name' key containing the endpoint name.
        The other fields of the returned dictionary and their types may vary across targets.

        Args:
            endpoint: Name of endpoint to fetch
        """
        raise NotImplementedError('This function is not implemented yet.')