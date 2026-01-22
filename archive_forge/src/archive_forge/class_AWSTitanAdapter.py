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
class AWSTitanAdapter(ProviderAdapter):

    @classmethod
    def completions_to_model(cls, payload, config):
        n = payload.pop('n', 1)
        if n != 1:
            raise HTTPException(status_code=422, detail=f"'n' must be '1' for AWS Titan models. Received value: '{n}'.")
        if 'temperature' in payload:
            payload['temperature'] = 0.5 * payload['temperature']
        return {'inputText': payload.pop('prompt'), 'textGenerationConfig': rename_payload_keys(payload, {'max_tokens': 'maxTokenCount', 'stop': 'stopSequences'})}

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(created=int(time.time()), object='text_completion', model=config.model.name, choices=[completions.Choice(index=idx, text=candidate.get('outputText'), finish_reason=None) for idx, candidate in enumerate(resp.get('results', []))], usage=completions.CompletionsUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError