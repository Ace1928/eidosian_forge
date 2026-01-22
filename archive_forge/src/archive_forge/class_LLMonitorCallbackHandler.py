import importlib.metadata
import logging
import os
import traceback
import warnings
from contextvars import ContextVar
from typing import Any, Dict, List, Union, cast
from uuid import UUID
import requests
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from packaging.version import parse
class LLMonitorCallbackHandler(BaseCallbackHandler):
    """Callback Handler for LLMonitor`.

    #### Parameters:
        - `app_id`: The app id of the app you want to report to. Defaults to
        `None`, which means that `LLMONITOR_APP_ID` will be used.
        - `api_url`: The url of the LLMonitor API. Defaults to `None`,
        which means that either `LLMONITOR_API_URL` environment variable
        or `https://app.llmonitor.com` will be used.

    #### Raises:
        - `ValueError`: if `app_id` is not provided either as an
        argument or as an environment variable.
        - `ConnectionError`: if the connection to the API fails.


    #### Example:
    ```python
    from langchain_community.llms import OpenAI
    from langchain_community.callbacks import LLMonitorCallbackHandler

    llmonitor_callback = LLMonitorCallbackHandler()
    llm = OpenAI(callbacks=[llmonitor_callback],
                 metadata={"userId": "user-123"})
    llm.invoke("Hello, how are you?")
    ```
    """
    __api_url: str
    __app_id: str
    __verbose: bool
    __llmonitor_version: str
    __has_valid_config: bool

    def __init__(self, app_id: Union[str, None]=None, api_url: Union[str, None]=None, verbose: bool=False) -> None:
        super().__init__()
        self.__has_valid_config = True
        try:
            import llmonitor
            self.__llmonitor_version = importlib.metadata.version('llmonitor')
            self.__track_event = llmonitor.track_event
        except ImportError:
            logger.warning('[LLMonitor] To use the LLMonitor callback handler you need to \n                have the `llmonitor` Python package installed. Please install it \n                with `pip install llmonitor`')
            self.__has_valid_config = False
            return
        if parse(self.__llmonitor_version) < parse('0.0.32'):
            logger.warning(f'[LLMonitor] The installed `llmonitor` version is \n                {self.__llmonitor_version} \n                but `LLMonitorCallbackHandler` requires at least version 0.0.32 \n                upgrade `llmonitor` with `pip install --upgrade llmonitor`')
            self.__has_valid_config = False
        self.__has_valid_config = True
        self.__api_url = api_url or os.getenv('LLMONITOR_API_URL') or DEFAULT_API_URL
        self.__verbose = verbose or bool(os.getenv('LLMONITOR_VERBOSE'))
        _app_id = app_id or os.getenv('LLMONITOR_APP_ID')
        if _app_id is None:
            logger.warning('[LLMonitor] app_id must be provided either as an argument or \n                as an environment variable')
            self.__has_valid_config = False
        else:
            self.__app_id = _app_id
        if self.__has_valid_config is False:
            return None
        try:
            res = requests.get(f'{self.__api_url}/api/app/{self.__app_id}')
            if not res.ok:
                raise ConnectionError()
        except Exception:
            logger.warning(f'[LLMonitor] Could not connect to the LLMonitor API at \n                {self.__api_url}')

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Union[UUID, None]=None, tags: Union[List[str], None]=None, metadata: Union[Dict[str, Any], None]=None, **kwargs: Any) -> None:
        if self.__has_valid_config is False:
            return
        try:
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)
            params = kwargs.get('invocation_params', {})
            params.update(serialized.get('kwargs', {}))
            name = params.get('model') or params.get('model_name') or params.get('model_id')
            if not name and 'anthropic' in params.get('_type'):
                name = 'claude-2'
            extra = {param: params.get(param) for param in PARAMS_TO_CAPTURE if params.get(param) is not None}
            input = _parse_input(prompts)
            self.__track_event('llm', 'start', user_id=user_id, run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, name=name, input=input, tags=tags, extra=extra, metadata=metadata, user_props=user_props, app_id=self.__app_id)
        except Exception as e:
            warnings.warn(f'[LLMonitor] An error occurred in on_llm_start: {e}')

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Union[UUID, None]=None, tags: Union[List[str], None]=None, metadata: Union[Dict[str, Any], None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)
            params = kwargs.get('invocation_params', {})
            params.update(serialized.get('kwargs', {}))
            name = params.get('model') or params.get('model_name') or params.get('model_id')
            if not name and 'anthropic' in params.get('_type'):
                name = 'claude-2'
            extra = {param: params.get(param) for param in PARAMS_TO_CAPTURE if params.get(param) is not None}
            input = _parse_lc_messages(messages[0])
            self.__track_event('llm', 'start', user_id=user_id, run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, name=name, input=input, tags=tags, extra=extra, metadata=metadata, user_props=user_props, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_chat_model_start: {e}')

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> None:
        if self.__has_valid_config is False:
            return
        try:
            token_usage = (response.llm_output or {}).get('token_usage', {})
            parsed_output: Any = [_parse_lc_message(generation.message) if hasattr(generation, 'message') else generation.text for generation in response.generations[0]]
            if len(parsed_output) == 1:
                parsed_output = parsed_output[0]
            self.__track_event('llm', 'end', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, output=parsed_output, token_usage={'prompt': token_usage.get('prompt_tokens'), 'completion': token_usage.get('completion_tokens')}, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_llm_end: {e}')

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, tags: Union[List[str], None]=None, metadata: Union[Dict[str, Any], None]=None, **kwargs: Any) -> None:
        if self.__has_valid_config is False:
            return
        try:
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)
            name = serialized.get('name')
            self.__track_event('tool', 'start', user_id=user_id, run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, name=name, input=input_str, tags=tags, metadata=metadata, user_props=user_props, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_tool_start: {e}')

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, tags: Union[List[str], None]=None, **kwargs: Any) -> None:
        output = str(output)
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event('tool', 'end', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, output=output, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_tool_end: {e}')

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Union[UUID, None]=None, tags: Union[List[str], None]=None, metadata: Union[Dict[str, Any], None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            name = serialized.get('id', [None, None, None, None])[3]
            type = 'chain'
            metadata = metadata or {}
            agentName = metadata.get('agent_name')
            if agentName is None:
                agentName = metadata.get('agentName')
            if name == 'AgentExecutor' or name == 'PlanAndExecute':
                type = 'agent'
            if agentName is not None:
                type = 'agent'
                name = agentName
            if parent_run_id is not None:
                type = 'chain'
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)
            input = _parse_input(inputs)
            self.__track_event(type, 'start', user_id=user_id, run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, name=name, input=input, tags=tags, metadata=metadata, user_props=user_props, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_chain_start: {e}')

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            output = _parse_output(outputs)
            self.__track_event('chain', 'end', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, output=output, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_chain_end: {e}')

    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            name = action.tool
            input = _parse_input(action.tool_input)
            self.__track_event('tool', 'start', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, name=name, input=input, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_agent_action: {e}')

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            output = _parse_output(finish.return_values)
            self.__track_event('agent', 'end', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, output=output, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_agent_finish: {e}')

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event('chain', 'error', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, error={'message': str(error), 'stack': traceback.format_exc()}, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_chain_error: {e}')

    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event('tool', 'error', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, error={'message': str(error), 'stack': traceback.format_exc()}, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_tool_error: {e}')

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Union[UUID, None]=None, **kwargs: Any) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event('llm', 'error', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None, error={'message': str(error), 'stack': traceback.format_exc()}, app_id=self.__app_id)
        except Exception as e:
            logger.error(f'[LLMonitor] An error occurred in on_llm_error: {e}')