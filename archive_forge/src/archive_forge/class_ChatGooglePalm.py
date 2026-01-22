from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
class ChatGooglePalm(BaseChatModel, BaseModel):
    """`Google PaLM` Chat models API.

    To use you must have the google.generativeai Python package installed and
    either:

        1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatGooglePalm
            chat = ChatGooglePalm()

    """
    client: Any
    model_name: str = 'models/chat-bison-001'
    'Model name to use.'
    google_api_key: Optional[SecretStr] = None
    temperature: Optional[float] = None
    'Run inference with this temperature. Must by in the closed\n       interval [0.0, 1.0].'
    top_p: Optional[float] = None
    'Decode using nucleus sampling: consider the smallest set of tokens whose\n       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0].'
    top_k: Optional[int] = None
    'Decode using top-k sampling: consider the set of top_k most probable tokens.\n       Must be positive.'
    n: int = 1
    'Number of chat completions to generate for each prompt. Note that the API may\n       not return the full n completions if duplicates are generated.'

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'google_api_key': 'GOOGLE_API_KEY'}

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'chat_models', 'google_palm']

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists, temperature, top_p, and top_k."""
        google_api_key = convert_to_secret_str(get_from_dict_or_env(values, 'google_api_key', 'GOOGLE_API_KEY'))
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key.get_secret_value())
        except ImportError:
            raise ChatGooglePalmError('Could not import google.generativeai python package. Please install it with `pip install google-generativeai`')
        values['client'] = genai
        if values['temperature'] is not None and (not 0 <= values['temperature'] <= 1):
            raise ValueError('temperature must be in the range [0.0, 1.0]')
        if values['top_p'] is not None and (not 0 <= values['top_p'] <= 1):
            raise ValueError('top_p must be in the range [0.0, 1.0]')
        if values['top_k'] is not None and values['top_k'] <= 0:
            raise ValueError('top_k must be positive')
        return values

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        prompt = _messages_to_prompt_dict(messages)
        response: genai.types.ChatResponse = chat_with_retry(self, model=self.model_name, prompt=prompt, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, candidate_count=self.n, **kwargs)
        return _response_to_result(response, stop)

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        prompt = _messages_to_prompt_dict(messages)
        response: genai.types.ChatResponse = await achat_with_retry(self, model=self.model_name, prompt=prompt, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, candidate_count=self.n)
        return _response_to_result(response, stop)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'model_name': self.model_name, 'temperature': self.temperature, 'top_p': self.top_p, 'top_k': self.top_k, 'n': self.n}

    @property
    def _llm_type(self) -> str:
        return 'google-palm-chat'