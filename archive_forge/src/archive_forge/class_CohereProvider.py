import json
import time
from typing import Any, AsyncGenerator, AsyncIterable, Dict
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
class CohereProvider(BaseProvider):
    NAME = 'Cohere'

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, CohereConfig):
            raise TypeError(f'Unexpected config type {config.model.config}')
        self.cohere_config: CohereConfig = config.model.config

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {self.cohere_config.cohere_api_key}'}

    @property
    def base_url(self) -> str:
        return 'https://api.cohere.ai/v1'

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(headers=self.auth_headers, base_url=self.base_url, path=path, payload=payload)

    def _stream_request(self, path: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        return send_stream_request(headers=self.auth_headers, base_url=self.base_url, path=path, payload=payload)

    async def chat_stream(self, payload: chat.RequestPayload) -> AsyncIterable[chat.StreamResponsePayload]:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = self._stream_request('chat', {'model': self.config.model.name, **CohereAdapter.chat_streaming_to_model(payload, self.config)})
        async for chunk in stream:
            if not chunk:
                continue
            resp = json.loads(chunk)
            if resp['event_type'] == 'stream-start':
                continue
            yield CohereAdapter.model_to_chat_streaming(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request('chat', {'model': self.config.model.name, **CohereAdapter.chat_to_model(payload, self.config)})
        return CohereAdapter.model_to_chat(resp, self.config)

    async def completions_stream(self, payload: completions.RequestPayload) -> AsyncIterable[completions.StreamResponsePayload]:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = self._stream_request('generate', {'model': self.config.model.name, **CohereAdapter.completions_streaming_to_model(payload, self.config)})
        async for chunk in stream:
            if not chunk:
                continue
            resp = json.loads(chunk)
            yield CohereAdapter.model_to_completions_streaming(resp, self.config)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request('generate', {'model': self.config.model.name, **CohereAdapter.completions_to_model(payload, self.config)})
        return CohereAdapter.model_to_completions(resp, self.config)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request('embed', {'model': self.config.model.name, **CohereAdapter.embeddings_to_model(payload, self.config)})
        return CohereAdapter.model_to_embeddings(resp, self.config)