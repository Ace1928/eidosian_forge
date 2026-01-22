import asyncio
import json
import warnings
from abc import ABC
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.anthropic import (
def _prepare_input_and_invoke_stream(self, prompt: Optional[str]=None, system: Optional[str]=None, messages: Optional[List[Dict]]=None, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
    _model_kwargs = self.model_kwargs or {}
    provider = self._get_provider()
    if stop:
        if provider not in self.provider_stop_sequence_key_name_map:
            raise ValueError(f'Stop sequence key name for {provider} is not supported.')
        _model_kwargs[self.provider_stop_sequence_key_name_map.get(provider)] = stop
    if provider == 'cohere':
        _model_kwargs['stream'] = True
    params = {**_model_kwargs, **kwargs}
    if self._guardrails_enabled:
        params.update(self._get_guardrails_canonical())
    input_body = LLMInputOutputAdapter.prepare_input(provider=provider, prompt=prompt, system=system, messages=messages, model_kwargs=params)
    body = json.dumps(input_body)
    request_options = {'body': body, 'modelId': self.model_id, 'accept': 'application/json', 'contentType': 'application/json'}
    if self._guardrails_enabled:
        request_options['guardrail'] = 'ENABLED'
        if self.guardrails.get('trace'):
            request_options['trace'] = 'ENABLED'
    try:
        response = self.client.invoke_model_with_response_stream(**request_options)
    except Exception as e:
        raise ValueError(f'Error raised by bedrock service: {e}')
    for chunk in LLMInputOutputAdapter.prepare_output_stream(provider, response, stop, True if messages else False):
        yield chunk
        self._get_bedrock_services_signal(chunk.generation_info)
        if run_manager is not None:
            run_manager.on_llm_new_token(chunk.text, chunk=chunk)