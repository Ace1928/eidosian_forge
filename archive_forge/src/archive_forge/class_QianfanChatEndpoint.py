import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
class QianfanChatEndpoint(BaseChatModel):
    """Baidu Qianfan chat models.

    To use, you should have the ``qianfan`` python package installed, and
    the environment variable ``qianfan_ak`` and ``qianfan_sk`` set with your
    API key and Secret Key.

    ak, sk are required parameters
    which you could get from  https://cloud.baidu.com/product/wenxinworkshop

    Example:
        .. code-block:: python

            from langchain_community.chat_models import QianfanChatEndpoint
            qianfan_chat = QianfanChatEndpoint(model="ERNIE-Bot",
                endpoint="your_endpoint", qianfan_ak="your_ak", qianfan_sk="your_sk")
    """
    init_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'init kwargs for qianfan client init, such as `query_per_second` which is \n        associated with qianfan resource object to limit QPS'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'extra params for model invoke using with `do`.'
    client: Any
    qianfan_ak: Optional[SecretStr] = None
    qianfan_sk: Optional[SecretStr] = None
    streaming: Optional[bool] = False
    'Whether to stream the results or not.'
    request_timeout: Optional[int] = Field(60, alias='timeout')
    'request timeout for chat http requests'
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1
    'Model params, only supported in ERNIE-Bot and ERNIE-Bot-turbo.\n    In the case of other model, passing these params will not affect the result.\n    '
    model: str = 'ERNIE-Bot-turbo'
    'Model name.\n    you could get from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu\n    \n    preset models are mapping to an endpoint.\n    `model` will be ignored if `endpoint` is set.\n    Default is ERNIE-Bot-turbo.\n    '
    endpoint: Optional[str] = None
    'Endpoint of the Qianfan LLM, required if custom model used.'

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values['qianfan_ak'] = convert_to_secret_str(get_from_dict_or_env(values, 'qianfan_ak', 'QIANFAN_AK', default=''))
        values['qianfan_sk'] = convert_to_secret_str(get_from_dict_or_env(values, 'qianfan_sk', 'QIANFAN_SK', default=''))
        params = {**values.get('init_kwargs', {}), 'model': values['model'], 'stream': values['streaming']}
        if values['qianfan_ak'].get_secret_value() != '':
            params['ak'] = values['qianfan_ak'].get_secret_value()
        if values['qianfan_sk'].get_secret_value() != '':
            params['sk'] = values['qianfan_sk'].get_secret_value()
        if values['endpoint'] is not None and values['endpoint'] != '':
            params['endpoint'] = values['endpoint']
        try:
            import qianfan
            values['client'] = qianfan.ChatCompletion(**params)
        except ImportError:
            raise ValueError('qianfan package not found, please install it with `pip install qianfan`')
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**{'endpoint': self.endpoint, 'model': self.model}, **super()._identifying_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat_model."""
        return 'baidu-qianfan-chat'

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Qianfan API."""
        normal_params = {'model': self.model, 'endpoint': self.endpoint, 'stream': self.streaming, 'request_timeout': self.request_timeout, 'top_p': self.top_p, 'temperature': self.temperature, 'penalty_score': self.penalty_score}
        return {**normal_params, **self.model_kwargs}

    def _convert_prompt_msg_params(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        """
        Converts a list of messages into a dictionary containing the message content
        and default parameters.

        Args:
            messages (List[BaseMessage]): The list of messages.
            **kwargs (Any): Optional arguments to add additional parameters to the
            resulting dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the message content and default
            parameters.

        """
        messages_dict: Dict[str, Any] = {'messages': [convert_message_to_dict(m) for m in messages if not isinstance(m, SystemMessage)]}
        for i in [i for i, m in enumerate(messages) if isinstance(m, SystemMessage)]:
            if 'system' not in messages_dict:
                messages_dict['system'] = ''
            messages_dict['system'] += cast(str, messages[i].content) + '\n'
        return {**messages_dict, **self._default_params, **kwargs}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        """Call out to an qianfan models endpoint for each generation with a prompt.
        Args:
            messages: The messages to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python
                response = qianfan_model("Tell me a joke.")
        """
        if self.streaming:
            completion = ''
            token_usage = {}
            chat_generation_info: Dict = {}
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                chat_generation_info = chunk.generation_info if chunk.generation_info is not None else chat_generation_info
                completion += chunk.text
            lc_msg = AIMessage(content=completion, additional_kwargs={})
            gen = ChatGeneration(message=lc_msg, generation_info=dict(finish_reason='stop'))
            return ChatResult(generations=[gen], llm_output={'token_usage': chat_generation_info.get('usage', {}), 'model_name': self.model})
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params['stop'] = stop
        response_payload = self.client.do(**params)
        lc_msg = _convert_dict_to_message(response_payload)
        gen = ChatGeneration(message=lc_msg, generation_info={'finish_reason': 'stop', **response_payload.get('body', {})})
        token_usage = response_payload.get('usage', {})
        llm_output = {'token_usage': token_usage, 'model_name': self.model}
        return ChatResult(generations=[gen], llm_output=llm_output)

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            completion = ''
            token_usage = {}
            chat_generation_info: Dict = {}
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                chat_generation_info = chunk.generation_info if chunk.generation_info is not None else chat_generation_info
                completion += chunk.text
            lc_msg = AIMessage(content=completion, additional_kwargs={})
            gen = ChatGeneration(message=lc_msg, generation_info=dict(finish_reason='stop'))
            return ChatResult(generations=[gen], llm_output={'token_usage': chat_generation_info.get('usage', {}), 'model_name': self.model})
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params['stop'] = stop
        response_payload = await self.client.ado(**params)
        lc_msg = _convert_dict_to_message(response_payload)
        generations = []
        gen = ChatGeneration(message=lc_msg, generation_info={'finish_reason': 'stop', **response_payload.get('body', {})})
        generations.append(gen)
        token_usage = response_payload.get('usage', {})
        llm_output = {'token_usage': token_usage, 'model_name': self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params['stop'] = stop
        params['stream'] = True
        for res in self.client.do(**params):
            if res:
                msg = _convert_dict_to_message(res)
                additional_kwargs = msg.additional_kwargs.get('function_call', {})
                chunk = ChatGenerationChunk(text=res['result'], message=AIMessageChunk(content=msg.content, role='assistant', additional_kwargs=additional_kwargs), generation_info=msg.additional_kwargs)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params['stop'] = stop
        params['stream'] = True
        async for res in await self.client.ado(**params):
            if res:
                msg = _convert_dict_to_message(res)
                additional_kwargs = msg.additional_kwargs.get('function_call', {})
                chunk = ChatGenerationChunk(text=res['result'], message=AIMessageChunk(content=msg.content, role='assistant', additional_kwargs=additional_kwargs), generation_info=msg.additional_kwargs)
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk