import time
from contextlib import contextmanager
from typing import Any, Dict, List
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import MosaicMLConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings
class MosaicMLProvider(BaseProvider):
    NAME = 'MosaicML'

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MosaicMLConfig):
            raise TypeError(f'Unexpected config type {config.model.config}')
        self.mosaicml_config: MosaicMLConfig = config.model.config

    async def _request(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {'Authorization': f'{self.mosaicml_config.mosaicml_api_key}'}
        return await send_request(headers=headers, base_url=self.mosaicml_config.mosaicml_api_base or 'https://models.hosted-on.mosaicml.hosting', path=model + '/v1/predict', payload=payload)

    @staticmethod
    def _parse_chat_messages_to_prompt(messages: List[chat.RequestMessage]) -> str:
        """
        This parser is based on the format described in
        https://huggingface.co/blog/llama2#how-to-prompt-llama-2 .
        The expected format is:
            "<s>[INST] <<SYS>>
            {{ system_prompt }}
            <</SYS>>

            {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>
            <s>[INST] {{ user_msg_2 }} [/INST]"
        """
        prompt = '<s>'
        for m in messages:
            if m.role == 'system' or m.role == 'user':
                inst = m.content
                if m.role == 'system':
                    inst = f'<<SYS>> {inst} <</SYS>>'
                inst += ' [/INST]'
                if prompt.endswith('[/INST]'):
                    prompt = prompt[:-7]
                else:
                    inst = f'[INST] {inst}'
                prompt += inst
            elif m.role == 'assistant':
                prompt += f' {m.content} </s><s>'
            else:
                raise MlflowException.invalid_parameter_value(f"Invalid role {m.role} inputted. Must be one of 'system', 'user', or 'assistant'.")
        if prompt.endswith('</s><s>'):
            prompt = prompt[:-7]
        return prompt

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        messages = payload.messages
        payload = jsonable_encoder(payload, exclude_none=True)
        payload.pop('messages', None)
        self.check_for_model_field(payload)
        key_mapping = {'max_tokens': 'max_new_tokens'}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(status_code=422, detail=f'Invalid parameter {k2}. Use {k1} instead.')
        payload = rename_payload_keys(payload, key_mapping)
        try:
            prompt = [self._parse_chat_messages_to_prompt(messages)]
        except MlflowException as e:
            raise HTTPException(status_code=422, detail=f'An invalid request structure was submitted. {e.message}')
        final_payload = {'inputs': prompt, 'parameters': payload}
        with custom_token_allowance_exceeded_handling():
            resp = await self._request(self.config.model.name, final_payload)
        return chat.ResponsePayload(created=int(time.time()), model=self.config.model.name, choices=[chat.Choice(index=idx, message=chat.ResponseMessage(role='assistant', content=c), finish_reason=None) for idx, c in enumerate(resp['outputs'])], usage=chat.ChatUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {'max_tokens': 'max_new_tokens'}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(status_code=422, detail=f'Invalid parameter {k2}. Use {k1} instead.')
        payload = rename_payload_keys(payload, key_mapping)
        prompt = payload.pop('prompt')
        if isinstance(prompt, str):
            prompt = [prompt]
        final_payload = {'inputs': prompt, 'parameters': payload}
        with custom_token_allowance_exceeded_handling():
            resp = await self._request(self.config.model.name, final_payload)
        return completions.ResponsePayload(created=int(time.time()), object='text_completion', model=self.config.model.name, choices=[completions.Choice(index=idx, text=c, finish_reason=None) for idx, c in enumerate(resp['outputs'])], usage=completions.CompletionsUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {'input': 'inputs'}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(status_code=422, detail=f'Invalid parameter {k2}. Use {k1} instead.')
        payload = rename_payload_keys(payload, key_mapping)
        if isinstance(payload['inputs'], str):
            payload['inputs'] = [payload['inputs']]
        resp = await self._request(self.config.model.name, payload)
        return embeddings.ResponsePayload(data=[embeddings.EmbeddingObject(embedding=output, index=idx) for idx, output in enumerate(resp['outputs'])], model=self.config.model.name, usage=embeddings.EmbeddingsUsage(prompt_tokens=None, total_tokens=None))