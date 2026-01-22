import logging
import os
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import (
from mlflow.utils.rest_utils import augmented_raise_for_status
from mlflow.deployments import BaseDeploymentClient
def _get_api_config_without_openai_dep() -> _OpenAIApiConfig:
    """
    Gets the parameters and configuration of the OpenAI API connected to.
    """
    api_type = os.getenv(_OpenAIEnvVar.OPENAI_API_TYPE.value)
    api_version = os.getenv(_OpenAIEnvVar.OPENAI_API_VERSION.value)
    api_base = os.getenv(_OpenAIEnvVar.OPENAI_API_BASE.value, None)
    engine = os.getenv(_OpenAIEnvVar.OPENAI_ENGINE.value, None)
    deployment_id = os.getenv(_OpenAIEnvVar.OPENAI_DEPLOYMENT_NAME.value, None)
    if api_type in ('azure', 'azure_ad', 'azuread'):
        batch_size = 16
        max_tokens_per_minute = 60000
    else:
        batch_size = 1024
        max_tokens_per_minute = 90000
    return _OpenAIApiConfig(api_type=api_type, batch_size=batch_size, max_requests_per_minute=3500, max_tokens_per_minute=max_tokens_per_minute, api_base=api_base, api_version=api_version, engine=engine, deployment_id=deployment_id)