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
class AI21Adapter(ProviderAdapter):

    @classmethod
    def completions_to_model(cls, payload, config):
        return rename_payload_keys(payload, {'stop': 'stopSequences', 'n': 'numResults', 'max_tokens': 'maxTokens'})

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(created=int(time.time()), object='text_completion', model=config.model.name, choices=[completions.Choice(index=idx, text=candidate.get('data', {}).get('text'), finish_reason=None) for idx, candidate in enumerate(resp.get('completions', []))], usage=completions.CompletionsUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError