import json
import time
from typing import AsyncIterable
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import AnthropicConfig, RouteConfig
from mlflow.gateway.constants import (
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions
class AnthropicAdapter(ProviderAdapter):

    @classmethod
    def chat_to_model(cls, payload, config):
        key_mapping = {'stop': 'stop_sequences'}
        payload = rename_payload_keys(payload, key_mapping)
        if 'top_p' in payload and 'temperature' in payload:
            raise HTTPException(status_code=422, detail="Cannot set both 'temperature' and 'top_p' parameters.")
        max_tokens = payload.get('max_tokens', MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS)
        if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
            raise HTTPException(status_code=422, detail=f'Invalid value for max_tokens: cannot exceed {MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.')
        payload['max_tokens'] = max_tokens
        if payload.pop('n', 1) != 1:
            raise HTTPException(status_code=422, detail="'n' must be '1' for the Anthropic provider. Received value: '{n}'.")
        system_messages = [m for m in payload['messages'] if m['role'] == 'system']
        if system_messages:
            payload['system'] = '\n'.join((m['content'] for m in system_messages))
        payload['messages'] = [m for m in payload['messages'] if m['role'] in ('user', 'assistant')]
        if 'temperature' in payload:
            payload['temperature'] = 0.5 * payload['temperature']
        return payload

    @classmethod
    def model_to_chat(cls, resp, config):
        stop_reason = 'length' if resp['stop_reason'] == 'max_tokens' else 'stop'
        return chat.ResponsePayload(id=resp['id'], created=int(time.time()), object='chat.completion', model=resp['model'], choices=[chat.Choice(index=0, message=chat.ResponseMessage(role='assistant', content=c['text']), finish_reason=stop_reason) for c in resp['content']], usage=chat.ChatUsage(prompt_tokens=resp['usage']['input_tokens'], completion_tokens=resp['usage']['output_tokens'], total_tokens=resp['usage']['input_tokens'] + resp['usage']['output_tokens']))

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        return cls.chat_to_model(payload, config)

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        content = resp.get('delta') or resp.get('content_block') or {}
        if (stop_reason := content.get('stop_reason')) is not None:
            stop_reason = 'length' if stop_reason == 'max_tokens' else 'stop'
        return chat.StreamResponsePayload(id=resp['id'], created=int(time.time()), model=resp['model'], choices=[chat.StreamChoice(index=resp['index'], finish_reason=stop_reason, delta=chat.StreamDelta(role=None, content=content.get('text')))])

    @classmethod
    def model_to_completions(cls, resp, config):
        stop_reason = 'stop' if resp['stop_reason'] == 'stop_sequence' else 'length'
        return completions.ResponsePayload(created=int(time.time()), object='text_completion', model=resp['model'], choices=[completions.Choice(index=0, text=resp['completion'], finish_reason=stop_reason)], usage=completions.CompletionsUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {'max_tokens': 'max_tokens_to_sample', 'stop': 'stop_sequences'}
        if 'top_p' in payload:
            raise HTTPException(status_code=422, detail="Cannot set both 'temperature' and 'top_p' parameters. Please use only the temperature parameter for your query.")
        max_tokens = payload.get('max_tokens', MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS)
        if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
            raise HTTPException(status_code=422, detail=f'Invalid value for max_tokens: cannot exceed {MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.')
        payload['max_tokens'] = max_tokens
        if payload.get('stream', False):
            raise HTTPException(status_code=422, detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow Gateway.")
        n = payload.pop('n', 1)
        if n != 1:
            raise HTTPException(status_code=422, detail=f"'n' must be '1' for the Anthropic provider. Received value: '{n}'.")
        payload = rename_payload_keys(payload, key_mapping)
        if payload['prompt'].startswith('Human: '):
            payload['prompt'] = '\n\n' + payload['prompt']
        if not payload['prompt'].startswith('\n\nHuman: '):
            payload['prompt'] = '\n\nHuman: ' + payload['prompt']
        if not payload['prompt'].endswith('\n\nAssistant:'):
            payload['prompt'] = payload['prompt'] + '\n\nAssistant:'
        if 'temperature' in payload:
            payload['temperature'] = 0.5 * payload['temperature']
        return payload

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError