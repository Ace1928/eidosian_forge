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
def _prepare_input_and_invoke(self, prompt: Optional[str]=None, system: Optional[str]=None, messages: Optional[List[Dict]]=None, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    _model_kwargs = self.model_kwargs or {}
    provider = self._get_provider()
    params = {**_model_kwargs, **kwargs}
    if self._guardrails_enabled:
        params.update(self._get_guardrails_canonical())
    input_body = LLMInputOutputAdapter.prepare_input(provider=provider, model_kwargs=params, prompt=prompt, system=system, messages=messages)
    body = json.dumps(input_body)
    accept = 'application/json'
    contentType = 'application/json'
    request_options = {'body': body, 'modelId': self.model_id, 'accept': accept, 'contentType': contentType}
    if self._guardrails_enabled:
        request_options['guardrail'] = 'ENABLED'
        if self.guardrails.get('trace'):
            request_options['trace'] = 'ENABLED'
    try:
        response = self.client.invoke_model(**request_options)
        text, body, usage_info = LLMInputOutputAdapter.prepare_output(provider, response).values()
    except Exception as e:
        raise ValueError(f'Error raised by bedrock service: {e}')
    if stop is not None:
        text = enforce_stop_tokens(text, stop)
    services_trace = self._get_bedrock_services_signal(body)
    if services_trace.get('signal') and run_manager is not None:
        run_manager.on_llm_error(Exception(f'Error raised by bedrock service: {services_trace.get('reason')}'), **services_trace)
    return (text, usage_info)