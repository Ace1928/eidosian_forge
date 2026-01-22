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
class AnthropicProvider(BaseProvider, AnthropicAdapter):
    NAME = 'Anthropic'

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AnthropicConfig):
            raise TypeError(f'Invalid config type {config.model.config}')
        self.anthropic_config: AnthropicConfig = config.model.config
        self.headers = {'x-api-key': self.anthropic_config.anthropic_api_key, 'anthropic-version': self.anthropic_config.anthropic_version}
        self.base_url = 'https://api.anthropic.com/v1/'

    async def chat_stream(self, payload: chat.RequestPayload) -> AsyncIterable[chat.StreamResponsePayload]:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = send_stream_request(headers=self.headers, base_url=self.base_url, path='messages', payload={'model': self.config.model.name, **AnthropicAdapter.chat_streaming_to_model(payload, self.config)})
        indices = []
        metadata = {}
        async for chunk in stream:
            chunk = chunk.strip()
            if not chunk:
                continue
            prefix, content = chunk.split(b':', 1)
            if prefix != b'data':
                continue
            resp = json.loads(content.decode('utf-8'))
            if resp['type'] == 'message_start':
                metadata['id'] = resp['message']['id']
                metadata['model'] = resp['message']['model']
                continue
            if resp['type'] not in ('message_delta', 'content_block_start', 'content_block_delta'):
                continue
            index = resp.get('index')
            if index is not None and index not in indices:
                indices.append(index)
            resp.update(metadata)
            if resp['type'] == 'message_delta':
                for index in indices:
                    yield AnthropicAdapter.model_to_chat_streaming({**resp, 'index': index}, self.config)
            else:
                yield AnthropicAdapter.model_to_chat_streaming(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(headers=self.headers, base_url=self.base_url, path='messages', payload={'model': self.config.model.name, **AnthropicAdapter.chat_to_model(payload, self.config)})
        return AnthropicAdapter.model_to_chat(resp, self.config)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(headers=self.headers, base_url=self.base_url, path='complete', payload={'model': self.config.model.name, **AnthropicAdapter.completions_to_model(payload, self.config)})
        return AnthropicAdapter.model_to_completions(resp, self.config)