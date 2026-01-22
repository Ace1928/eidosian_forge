from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from langchain_core.utils.env import get_from_dict_or_env
from langchain_community.adapters.openai import convert_message_to_dict
@deprecated(since='0.0.26', removal='0.2', alternative_import='langchain_fireworks.ChatFireworks')
class ChatFireworks(BaseChatModel):
    """Fireworks Chat models."""
    model: str = 'accounts/fireworks/models/llama-v2-7b-chat'
    model_kwargs: dict = Field(default_factory=lambda: {'temperature': 0.7, 'max_tokens': 512, 'top_p': 1}.copy())
    fireworks_api_key: Optional[SecretStr] = None
    max_retries: int = 20
    use_retry: bool = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'fireworks_api_key': 'FIREWORKS_API_KEY'}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'chat_models', 'fireworks']

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        try:
            import fireworks.client
        except ImportError as e:
            raise ImportError('Could not import fireworks-ai python package. Please install it with `pip install fireworks-ai`.') from e
        fireworks_api_key = convert_to_secret_str(get_from_dict_or_env(values, 'fireworks_api_key', 'FIREWORKS_API_KEY'))
        fireworks.client.api_key = fireworks_api_key.get_secret_value()
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'fireworks-chat'

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)
        params = {'model': self.model, 'messages': message_dicts, **self.model_kwargs, **kwargs}
        response = completion_with_retry(self, self.use_retry, run_manager=run_manager, stop=stop, **params)
        return self._create_chat_result(response)

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)
        params = {'model': self.model, 'messages': message_dicts, **self.model_kwargs, **kwargs}
        response = await acompletion_with_retry(self, self.use_retry, run_manager=run_manager, stop=stop, **params)
        return self._create_chat_result(response)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        if llm_outputs[0] is None:
            return {}
        return llm_outputs[0]

    def _create_chat_result(self, response: Any) -> ChatResult:
        generations = []
        for res in response.choices:
            message = convert_dict_to_message(res.message)
            gen = ChatGeneration(message=message, generation_info=dict(finish_reason=res.finish_reason))
            generations.append(gen)
        llm_output = {'model': self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages)
        default_chunk_class = AIMessageChunk
        params = {'model': self.model, 'messages': message_dicts, 'stream': True, **self.model_kwargs, **kwargs}
        for chunk in completion_with_retry(self, self.use_retry, run_manager=run_manager, stop=stop, **params):
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)
            finish_reason = choice.finish_reason
            generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages)
        default_chunk_class = AIMessageChunk
        params = {'model': self.model, 'messages': message_dicts, 'stream': True, **self.model_kwargs, **kwargs}
        async for chunk in await acompletion_with_retry_streaming(self, self.use_retry, run_manager=run_manager, stop=stop, **params):
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)
            finish_reason = choice.finish_reason
            generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.text, chunk=cg_chunk)
            yield cg_chunk