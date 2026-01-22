from __future__ import annotations
import logging
import os
import sys
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs
from langchain_community.utils.openai import is_openai_v1
@deprecated(since='0.0.1', removal='0.2.0', alternative_import='langchain_openai.ChatOpenAI')
class OpenAIChat(BaseLLM):
    """OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.llms import OpenAIChat
            openaichat = OpenAIChat(model_name="gpt-3.5-turbo")
    """
    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)
    model_name: str = 'gpt-3.5-turbo'
    'Model name to use.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Holds any model parameters valid for `create` call not explicitly specified.'
    openai_api_key: Optional[str] = Field(default=None, alias='api_key')
    'Automatically inferred from env var `OPENAI_API_KEY` if not provided.'
    openai_api_base: Optional[str] = Field(default=None, alias='base_url')
    'Base URL path for API requests, leave blank if not using a proxy or service \n        emulator.'
    openai_proxy: Optional[str] = None
    max_retries: int = 6
    'Maximum number of retries to make when generating.'
    prefix_messages: List = Field(default_factory=list)
    'Series of messages for Chat input.'
    streaming: bool = False
    'Whether to stream the results or not.'
    allowed_special: Union[Literal['all'], AbstractSet[str]] = set()
    'Set of special tokens that are allowed。'
    disallowed_special: Union[Literal['all'], Collection[str]] = 'all'
    'Set of special tokens that are not allowed。'

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}
        extra = values.get('model_kwargs', {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f'Found {field_name} supplied twice.')
                extra[field_name] = values.pop(field_name)
        values['model_kwargs'] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(values, 'openai_api_key', 'OPENAI_API_KEY')
        openai_api_base = get_from_dict_or_env(values, 'openai_api_base', 'OPENAI_API_BASE', default='')
        openai_proxy = get_from_dict_or_env(values, 'openai_proxy', 'OPENAI_PROXY', default='')
        openai_organization = get_from_dict_or_env(values, 'openai_organization', 'OPENAI_ORGANIZATION', default='')
        try:
            import openai
            openai.api_key = openai_api_key
            if openai_api_base:
                openai.api_base = openai_api_base
            if openai_organization:
                openai.organization = openai_organization
            if openai_proxy:
                openai.proxy = {'http': openai_proxy, 'https': openai_proxy}
        except ImportError:
            raise ImportError('Could not import openai python package. Please install it with `pip install openai`.')
        try:
            values['client'] = openai.ChatCompletion
        except AttributeError:
            raise ValueError('`openai` has no `ChatCompletion` attribute, this is likely due to an old version of the openai package. Try upgrading it with `pip install --upgrade openai`.')
        warnings.warn('You are trying to use a chat model. This way of initializing it is no longer supported. Instead, please use: `from langchain_community.chat_models import ChatOpenAI`')
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return self.model_kwargs

    def _get_chat_params(self, prompts: List[str], stop: Optional[List[str]]=None) -> Tuple:
        if len(prompts) > 1:
            raise ValueError(f'OpenAIChat currently only supports single prompt, got {prompts}')
        messages = self.prefix_messages + [{'role': 'user', 'content': prompts[0]}]
        params: Dict[str, Any] = {**{'model': self.model_name}, **self._default_params}
        if stop is not None:
            if 'stop' in params:
                raise ValueError('`stop` found in both the input and default params.')
            params['stop'] = stop
        if params.get('max_tokens') == -1:
            del params['max_tokens']
        return (messages, params)

    def _stream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
        messages, params = self._get_chat_params([prompt], stop)
        params = {**params, **kwargs, 'stream': True}
        for stream_resp in completion_with_retry(self, messages=messages, run_manager=run_manager, **params):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            token = stream_resp['choices'][0]['delta'].get('content', '')
            chunk = GenerationChunk(text=token)
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    async def _astream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        messages, params = self._get_chat_params([prompt], stop)
        params = {**params, **kwargs, 'stream': True}
        async for stream_resp in await acompletion_with_retry(self, messages=messages, run_manager=run_manager, **params):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            token = stream_resp['choices'][0]['delta'].get('content', '')
            chunk = GenerationChunk(text=token)
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def _generate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])
        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        full_response = completion_with_retry(self, messages=messages, run_manager=run_manager, **params)
        if not isinstance(full_response, dict):
            full_response = full_response.dict()
        llm_output = {'token_usage': full_response['usage'], 'model_name': self.model_name}
        return LLMResult(generations=[[Generation(text=full_response['choices'][0]['message']['content'])]], llm_output=llm_output)

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            async for chunk in self._astream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])
        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        full_response = await acompletion_with_retry(self, messages=messages, run_manager=run_manager, **params)
        if not isinstance(full_response, dict):
            full_response = full_response.dict()
        llm_output = {'token_usage': full_response['usage'], 'model_name': self.model_name}
        return LLMResult(generations=[[Generation(text=full_response['choices'][0]['message']['content'])]], llm_output=llm_output)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{'model_name': self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'openai-chat'

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs using the tiktoken package."""
        if sys.version_info[1] < 8:
            return super().get_token_ids(text)
        try:
            import tiktoken
        except ImportError:
            raise ImportError('Could not import tiktoken python package. This is needed in order to calculate get_num_tokens. Please install it with `pip install tiktoken`.')
        enc = tiktoken.encoding_for_model(self.model_name)
        return enc.encode(text, allowed_special=self.allowed_special, disallowed_special=self.disallowed_special)