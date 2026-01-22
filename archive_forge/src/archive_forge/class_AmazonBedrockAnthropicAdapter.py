import json
import time
from enum import Enum
import boto3
import botocore.config
import botocore.exceptions
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import AmazonBedrockConfig, AWSIdAndKey, AWSRole, RouteConfig
from mlflow.gateway.constants import (
from mlflow.gateway.exceptions import AIGatewayConfigException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys
from mlflow.gateway.schemas import completions
class AmazonBedrockAnthropicAdapter(AnthropicAdapter):

    @classmethod
    def completions_to_model(cls, payload, config):
        payload = super().completions_to_model(payload, config)
        if '\n\nHuman:' not in payload.get('stop_sequences', []):
            payload.setdefault('stop_sequences', []).append('\n\nHuman:')
        payload['max_tokens_to_sample'] = min(payload.get('max_tokens_to_sample', MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS), AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS)
        return payload

    @classmethod
    def model_to_completions(cls, payload, config):
        payload['model'] = config.model.name
        return super().model_to_completions(payload, config)